"""CatFlow adapter: native transformer, explicit probability path and raw export."""
import copy
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from common import (read_job, load_split, torch_load, save_checkpoint, finish,
                    export_samples, finite_loss, selective_module)
from catflow_path import training_interpolant, probe_upstream_path, path_record


def upstream(job):
    root = Path(job["source_root"])
    utilities = selective_module("utils", root / "utils.py",
        selected={"PlaceHolder", "assert_correctly_masked", "get_GT_model", "get_tau_sched"},
        globals_dict={"torch": torch, "nn": torch.nn})
    flow = selective_module("flow_matching", root / "flow_matching.py",
        selected={"conditional_velocity", "hyperplane_proj"}, globals_dict={"torch": torch})
    return utilities, flow


def ema_class(backend=None):
    """Use the native bundled EMA when the optional torch_ema package is absent.

    Record the backend in the checkpoint so reload does not silently change it.
    Both implement the source's update-count-adjusted exponential average.
    """
    if backend in (None, "torch_ema"):
        try:
            from torch_ema import ExponentialMovingAverage
            return ExponentialMovingAverage, "torch_ema"
        except ModuleNotFoundError as exc:
            if backend == "torch_ema" or exc.name != "torch_ema":
                raise
    if backend not in (None, "bundled"):
        raise ValueError("Unknown EMA backend: " + str(backend))
    from ema import ExponentialMovingAverage
    return ExponentialMovingAverage, "bundled"


def dense_batch(data, indices, device, dx, de):
    a = torch.as_tensor(data["adjacency"][indices], device=device).long()
    types = torch.as_tensor(data["node_types"][indices], device=device).long()
    n = torch.as_tensor(data["num_nodes"][indices], device=device).long()
    mask = torch.arange(a.size(1), device=device)[None, :] < n[:, None]
    pair_mask = mask[:, :, None] & mask[:, None, :]
    pair_mask &= ~torch.eye(a.size(1), dtype=torch.bool, device=device)[None]
    x = F.one_hot(types.clamp(min=0), dx).float() * mask[:, :, None]
    e = F.one_hot(a, de).float() * pair_mask[:, :, :, None]
    return x, e, mask, pair_mask


def cpu_snapshot(value):
    """Detach snapshots so best-checkpoint state cannot alias live parameters."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return copy.deepcopy(value)


def batch_loss(model, data, ids, device, dx, de, flow, path):
    x1, e1, mask, pair_mask = dense_batch(data, ids, device, dx, de)
    t = torch.rand(len(ids), 1, 1, device=device)
    xt = training_interpolant(x1, t, path, flow)
    et = training_interpolant(e1, t.unsqueeze(-1), path, flow, edge=True)
    diag = torch.eye(et.size(1), dtype=torch.bool, device=device)[None]
    et = et.masked_fill(diag[..., None], 0)
    pred = model(xt, et, t.reshape(len(ids), 1), mask)
    lx = F.cross_entropy(pred.X[mask], x1.argmax(-1)[mask])
    le = (F.cross_entropy(pred.E[pair_mask], e1.argmax(-1)[pair_mask])
          if pair_mask.any() else pred.E.sum() * 0)
    loss = lx + 5 * le
    finite_loss(loss, "CatFlow loss")
    return loss, float(lx.item()), float(le.item())


def validation_loss(model, ema, data, device, dx, de, flow, path, batch_size, repeats, seed):
    """Evaluate EMA weights on fixed noise/time draws from validation only.

    Fixed RNG is isolated from training. EMA parameters are restored even on
    errors; both torch-ema and the uploaded fallback implement store/copy/restore.
    """
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    was_training = model.training
    ema.store(model.parameters())
    try:
        ema.copy_to(model.parameters())
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            total, count = 0., 0
            for _ in range(repeats):
                for start in range(0, len(data["num_nodes"]), batch_size):
                    ids = np.arange(start, min(start + batch_size, len(data["num_nodes"])))
                    loss, _, _ = batch_loss(model, data, ids, device, dx, de, flow, path)
                    total += float(loss.item()) * len(ids)
                    count += len(ids)
        if count == 0:
            raise ValueError("CatFlow requires a nonempty validation split.")
        return total / count
    finally:
        ema.restore(model.parameters())
        model.train(was_training)


def select_weights(state, sample):
    selection = sample.get("checkpoint_selection", "auto")
    if selection not in {"auto", "best", "last"}:
        raise ValueError("sample.checkpoint_selection must be auto, best or last.")
    if selection == "auto":
        selection = "best" if state.get("best") is not None else "last"
    if selection == "best":
        if state.get("best") is None:
            raise ValueError("This legacy checkpoint has no best snapshot; select last or retrain with the v2 path.")
        return state["best"], selection
    return state, selection


def main():
    job, device = read_job()
    utils, flow = upstream(job)
    options = job["options"]
    spec = options.get("model", {})
    if set(spec) - {"num_layers", "small_model", "task"}:
        raise ValueError("Unknown CatFlow model options: " + str(set(spec) - {"num_layers", "small_model", "task"}))
    args = SimpleNamespace(num_layers=int(spec.get("num_layers", 6)), small_model=int(spec.get("small_model", 0)),
                           task=spec.get("task", "abstract" if job["profile"]["domain"] == "generic" else job["profile"]["benchmark_id"]))
    dx = len(job["profile"]["atomic_numbers"]) or 1
    de = len(job["profile"]["bond_types"]) + 1 if dx > 1 else 2
    model = utils.get_GT_model(args, dx, de).to(device)
    train = load_split(job)
    cfg = options["train"]
    if cfg.get("distribution", "normal") != "normal" or cfg.get("loss_function", "kld") != "kld":
        raise ValueError("The CatFlow adapter supports normal/kld; train.path explicitly selects the probability path.")
    if job["stage"] == "train":
        path = cfg.get("path", "linear")
        probe = probe_upstream_path(flow)
        probability_path = path_record(path, job["source_root"], probe)
        print("CatFlow probability path: " + json.dumps(probability_path, sort_keys=True), flush=True)
        if path == "upstream" and probe["node_residual_std_at_t1"] > 1e-4:
            warnings.warn("UPSTREAM PATH HAS NONVANISHING ENDPOINT NOISE; the straight-line sampler does not match it. "
                          "This mode is for labelled legacy diagnostics, not the corrected CatFlow baseline.")
        allowed_train = {"epochs", "batch_size", "lr", "ema", "log_every", "distribution", "loss_function", "path",
                         "validation_every", "validation_repeats", "validation_seed", "checkpoint_every"}
        if set(cfg) - allowed_train:
            raise ValueError("Unknown CatFlow train options: " + str(set(cfg) - allowed_train))
        ExponentialMovingAverage, ema_backend = ema_class()
        val = load_split(job, "val")
        epochs, batch_size = int(cfg["epochs"]), int(cfg["batch_size"])
        val_every = int(cfg.get("validation_every", 100))
        val_repeats = int(cfg.get("validation_repeats", 4))
        save_every = int(cfg.get("checkpoint_every", 1000))
        if min(val_every, val_repeats, save_every) < 1:
            raise ValueError("Validation/checkpoint intervals and validation repeats must be positive.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-4)), weight_decay=1e-12)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.get("ema", .999)))
        history, best, optimizer_steps = [], None, 0
        history_path = Path(job["checkpoint"]).parent / "training_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text("")
        def checkpoint(epoch):
            save_checkpoint(job["checkpoint"], {"format_version": 2, "model": model.state_dict(), "ema": ema.state_dict(),
                "ema_backend": ema_backend, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "trained_epochs": epoch, "optimizer_steps": optimizer_steps, "history": history,
                "best": best, "checkpoint_selection": "best_val_ema", "probability_path": probability_path,
                "architecture": vars(args), "dx": dx, "de": de})
        for epoch in range(1, epochs + 1):
            model.train()
            indices = np.random.permutation(len(train["num_nodes"]))
            total, total_x, total_e, count = 0., 0., 0., 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for start in range(0, len(indices), batch_size):
                ids = indices[start:start + batch_size]
                loss, lx, le = batch_loss(model, train, ids, device, dx, de, flow, path)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                ema.update(model.parameters())
                optimizer_steps += 1
                total += loss.item() * len(ids)
                total_x += lx * len(ids)
                total_e += le * len(ids)
                count += len(ids)
            if count == 0:
                raise ValueError("CatFlow requires a nonempty training split.")
            row = {"epoch": epoch, "optimizer_steps": optimizer_steps, "lr": learning_rate,
                   "train": total / count, "train_x": total_x / count, "train_e": total_e / count}
            if epoch == 1 or epoch % val_every == 0 or epoch == epochs:
                score = validation_loss(model, ema, val, device, dx, de, flow, path, batch_size,
                                        val_repeats, int(cfg.get("validation_seed", job["seed"] + 1729)))
                row["val"] = row["val_ema"] = score
                if best is None or score < best["val_ema"]:
                    best = {"model": cpu_snapshot(model.state_dict()), "ema": cpu_snapshot(ema.state_dict()),
                            "epoch": epoch, "val_ema": score}
            scheduler.step()
            history.append(row)
            with history_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            if epoch % int(cfg.get("log_every", 100)) == 0 or epoch in (1, epochs):
                print("CatFlow epoch %d/%d steps=%d train=%.6f val_ema=%s best_epoch=%d lr=%.3g" %
                      (epoch, epochs, optimizer_steps, row["train"], row.get("val_ema", "not_evaluated"), best["epoch"], learning_rate), flush=True)
            if epoch % save_every == 0 or epoch == epochs:
                checkpoint(epoch)
        finish(job, {"epochs": epochs, "optimizer_steps": optimizer_steps, "ema_backend": ema_backend,
                     "validation_split": "val", "validation_weights": "ema", "fixed_validation_draws": True,
                     "validation_repeats": val_repeats, "checkpoint_selection": "best_val_ema", "best_epoch": best["epoch"],
                     "best_val_ema": best["val_ema"], "probability_path": probability_path})
    else:
        state = torch_load(job["checkpoint"], device)
        if state.get("dx", dx) != dx or state.get("de", de) != de or state.get("architecture", vars(args)) != vars(args):
            raise ValueError("CatFlow checkpoint architecture/categories do not match the managed run.")
        sample = options.get("sample", {})
        allowed_sample = {"use_ema", "checkpoint_selection", "method", "steps", "t_end", "atol", "rtol"}
        if set(sample) - allowed_sample:
            raise ValueError("Unknown CatFlow sample options: " + str(set(sample) - allowed_sample))
        weights, selection = select_weights(state, sample)
        model.load_state_dict(weights["model"])
        if state.get("probability_path") is None:
            warnings.warn("LEGACY CHECKPOINT: trained via upstream conditional_velocity; probability path is unrecorded. "
                          "Generation retains the old velocity. Changing a training YAML cannot repair these weights.")
        if sample.get("use_ema", True):
            ExponentialMovingAverage, _ = ema_class(state.get("ema_backend", "torch_ema"))
            ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.get("ema", .999)))
            ema.load_state_dict(weights["ema"])
            # copy_to handles CPU best snapshots with a GPU model too; unlike .to,
            # this also works with the source EMA's list-backed shadow parameters.
            ema.copy_to(model.parameters())
        print("CatFlow sampling %s %s weights from epoch %s" %
              (selection, "EMA" if sample.get("use_ema", True) else "raw", weights.get("epoch", "unknown")), flush=True)
        model.eval()
        nmax = job["profile"]["max_nodes"]
        all_a, all_x, all_n = [], [], []
        batch_size = int(options.get("generation_batch_size", 128))
        endpoint = float(sample.get("t_end", .95))
        if not 0 < endpoint < 1:
            raise ValueError("CatFlow kld velocity is singular at t=1; require 0 < sample.t_end < 1.")
        method = sample.get("method", "dopri5")
        steps = int(sample.get("steps", 100))
        if steps < 1:
            raise ValueError("sample.steps must be positive.")
        nfe = 0
        for start in range(0, job["num_graphs"], batch_size):
            b = min(batch_size, job["num_graphs"] - start)
            sizes = np.random.choice(train["num_nodes"], b, replace=True)
            mask = torch.arange(nmax, device=device)[None] < torch.as_tensor(sizes, device=device)[:, None]
            pair_mask = mask[:, :, None] & mask[:, None, :]
            pair_mask &= ~torch.eye(nmax, dtype=torch.bool, device=device)[None]
            x = torch.randn(b, nmax, dx, device=device)
            e = torch.randn(b, nmax, nmax, de, device=device)
            e = (e + e.transpose(1, 2)) / 2
            x = x * mask[..., None]
            e = e * pair_mask[..., None]
            def velocity(t, values):
                nonlocal nfe
                nfe += 1
                x, e = values
                pred = model(x, e * pair_mask[..., None], torch.ones(b, 1, device=device) * t, mask)
                return ((pred.X.softmax(-1) - x) / (1 - t) * mask[..., None],
                        (pred.E.softmax(-1) - e) / (1 - t) * pair_mask[..., None])
            with torch.no_grad():
                if method == "euler":
                    dt = endpoint / steps
                    for step in range(steps):
                        vx, ve = velocity(step * dt, (x, e))
                        x, e = x + dt * vx, e + dt * ve
                else:
                    from torchdiffeq import odeint
                    solver_options = {"step_size": endpoint / steps} if method in {"midpoint", "rk4", "explicit_adams", "implicit_adams"} else None
                    xx, ee = odeint(velocity, (x, e), torch.tensor([0., endpoint], device=device), method=method,
                                    atol=float(sample.get("atol", 1e-5)), rtol=float(sample.get("rtol", 1e-5)), options=solver_options)
                    x, e = xx[-1], ee[-1]
            if not torch.isfinite(x).all() or not torch.isfinite(e).all():
                raise FloatingPointError("CatFlow ODE produced non-finite states.")
            e = (e + e.transpose(1, 2)) / 2
            all_a.append(e.argmax(-1).cpu().numpy())
            all_x.append(x.argmax(-1).cpu().numpy())
            all_n.extend(sizes.tolist())
            print("CatFlow generated %d/%d" % (len(all_n), job["num_graphs"]), flush=True)
        export_samples(job, np.concatenate(all_a), np.asarray(all_n, dtype=np.int64), np.concatenate(all_x))
        record_path = Path(job["worker_manifest"])
        record = json.loads(record_path.read_text())
        record.update(probability_path=state.get("probability_path", {"name": "legacy_upstream_unrecorded"}),
                      checkpoint_selection=selection, sampled_epoch=weights.get("epoch"),
                      use_ema=bool(sample.get("use_ema", True)), solver=method, t_end=endpoint, total_nfe=nfe)
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
