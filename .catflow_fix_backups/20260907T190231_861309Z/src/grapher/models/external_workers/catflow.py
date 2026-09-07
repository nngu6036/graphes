"""CatFlow adapter: native transformer/path, managed dense batches and raw export."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from common import (read_job, load_split, torch_load, save_checkpoint, finish,
                    export_samples, finite_loss, selective_module)


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
    # The shipped normal conditional path includes the source's extra noise.
    # Do not silently replace it by another flow-matching formulation.
    distribution = cfg.get("distribution", "normal")
    if distribution != "normal" or cfg.get("loss_function", "kld") != "kld":
        raise ValueError("This CatFlow adapter supports the supplied default normal/kld path only.")
    if job["stage"] == "train":
        ExponentialMovingAverage, ema_backend = ema_class()
        val = load_split(job, "val")
        epochs, batch_size = int(cfg["epochs"]), int(cfg["batch_size"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 2e-4)), weight_decay=1e-12)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.get("ema", .999)))
        history = []
        for epoch in range(epochs):
            totals = {}
            for split, data in (("train", train), ("val", val)):
                model.train(split == "train")
                indices = np.random.permutation(len(data["num_nodes"])) if split == "train" else np.arange(len(data["num_nodes"]))
                total, count = 0., 0
                with torch.set_grad_enabled(split == "train"):
                    for start in range(0, len(indices), batch_size):
                        ids = indices[start:start + batch_size]
                        x1, e1, mask, pair_mask = dense_batch(data, ids, device, dx, de)
                        t = torch.rand(len(ids), 1, 1, device=device)
                        xt, _ = flow.conditional_velocity("normal", x1, t, None, 8, 0)
                        et, _ = flow.conditional_velocity("normal", e1, t.unsqueeze(-1), None, 8, 0, edge=True)
                        # Native training zeros only the diagonal before the
                        # transformer; preserve this (padding is masked inside).
                        diag = torch.eye(et.size(1), dtype=torch.bool, device=device)[None].expand(len(ids), -1, -1)
                        et = et.masked_fill(diag[..., None], 0)
                        pred = model(xt, et, t.reshape(len(ids), 1), mask)
                        loss_x = F.cross_entropy(pred.X[mask], x1.argmax(-1)[mask])
                        loss_e = (F.cross_entropy(pred.E[pair_mask], e1.argmax(-1)[pair_mask])
                                  if pair_mask.any() else pred.E.sum() * 0)
                        loss = loss_x + 5 * loss_e
                        finite_loss(loss, "CatFlow loss")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                            ema.update(model.parameters())
                        total += loss.item() * len(ids)
                        count += len(ids)
                totals[split] = total / count
            scheduler.step()
            history.append({"epoch": epoch + 1, **totals})
            if (epoch + 1) % int(cfg.get("log_every", 1)) == 0 or epoch + 1 == epochs:
                print("CatFlow epoch %d/%d train=%.6f val=%.6f" % (epoch + 1, epochs, totals["train"], totals["val"]), flush=True)
        save_checkpoint(job["checkpoint"], {"model": model.state_dict(), "ema": ema.state_dict(), "ema_backend": ema_backend,
                        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "epoch": epochs, "history": history, "architecture": vars(args), "dx": dx, "de": de})
        finish(job, {"epochs": epochs, "ema_backend": ema_backend, "validation_split": "val", "checkpoint_selection": "final_epoch"})
    else:
        state = torch_load(job["checkpoint"], device)
        model.load_state_dict(state["model"])
        sample = options.get("sample", {})
        if sample.get("use_ema", True):
            ExponentialMovingAverage, _ = ema_class(state.get("ema_backend", "torch_ema"))
            ema = ExponentialMovingAverage(model.parameters(), decay=float(cfg.get("ema", .999)))
            ema.load_state_dict(state["ema"])
            ema.to(device)
            ema.copy_to(model.parameters())
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
                    xx, ee = odeint(velocity, (x, e), torch.tensor([0., endpoint], device=device), method=method,
                                    atol=float(sample.get("atol", 1e-5)), rtol=float(sample.get("rtol", 1e-5)))
                    x, e = xx[-1], ee[-1]
            if not torch.isfinite(x).all() or not torch.isfinite(e).all():
                raise FloatingPointError("CatFlow ODE produced non-finite states.")
            e = (e + e.transpose(1, 2)) / 2
            all_a.append(e.argmax(-1).cpu().numpy())
            all_x.append(x.argmax(-1).cpu().numpy())
            all_n.extend(sizes.tolist())
            print("CatFlow generated %d/%d" % (len(all_n), job["num_graphs"]), flush=True)
        export_samples(job, np.concatenate(all_a), all_n, np.concatenate(all_x))


if __name__ == "__main__":
    main()
