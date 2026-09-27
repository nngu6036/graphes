"""Standalone Option-C training; never calls a GDSM wrapper or legacy runner."""
from __future__ import annotations

import copy
import json
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from . import CHECKPOINT_FORMAT
from .config import generation_contract, resume_contract, validate_config
from .data import load_data, collate, permute_aligned, TypedGraphletsMulti
from .diffusion import MarginalNoise, cosine_alpha_bar, draw_categories, symmetric_noise, q_sample
from .losses import joint_loss
from .model import OptionCDenoiser, model_config
from .runtime import (cpu_tree, load_checkpoint, resolve_device, save_torch, seed_everything,
                      validate_output_dir, versions, write_json, atomic_write, source_fingerprint)

TRAIN_FORMAT = "grapher_option_c_training_v1"


def loss_batch(model, rows, basis, cfg, schedule, node_noise, generator, device, *, augment):
    batch = collate(rows, basis, device=device)
    if augment:
        batch = permute_aligned(batch, generator)
    t = torch.randint(1, len(schedule), (len(rows),), device=device, generator=generator)
    xt = draw_categories(node_noise.forward_probs(batch["x"], t), generator).masked_fill(~batch["mask"], 0)
    eps = symmetric_noise(batch["mask"], generator)
    wt = q_sample(batch["w"], eps, schedule, t, batch["mask"])
    pred = model(xt, wt, t, batch["mask"], len(schedule)-1)
    return joint_loss(pred, batch, basis, cfg["loss_weights"], cfg["spectral"])


def _run_epoch(model, rows, basis, cfg, schedule, node_noise, generator, device,
               *, order, optimizer=None, ema_state=None, epoch=0):
    training = optimizer is not None
    model.train(training)
    batch_size = cfg["training"]["batch_size"]
    sums, count, steps = defaultdict(float), 0, 0
    started = time.monotonic()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(order), batch_size):
            selected = [rows[int(i)] for i in order[start:start+batch_size]]
            loss, parts = loss_batch(model, selected, basis, cfg, schedule, node_noise, generator, device,
                                     augment=training and cfg["training"]["permutation_augmentation"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {'training' if training else 'validation'} loss at epoch {epoch}")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                if ema_state is not None:
                    decay = cfg["training"]["ema_decay"]
                    with torch.no_grad():
                        for key, value in model.state_dict().items():
                            if value.is_floating_point():
                                ema_state[key].mul_(decay).add_(value, alpha=1-decay)
                            else:
                                ema_state[key].copy_(value)
                steps += 1
            for key, value in {"loss": loss, **parts}.items():
                sums[key] += float(value.detach()) * len(selected)
            count += len(selected)
            every = cfg["training"]["log_every_batches"]
            if training and every and steps % every == 0:
                print(f"[option-c] epoch={epoch} batch={steps} graphs={count}/{len(order)} "
                      f"loss={sums['loss']/count:.6f} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    return {key: value/count for key, value in sums.items()}, steps


def train(cfg, output_dir, *, seed, device="auto", resume=False, overwrite=False):
    cfg = validate_config(cfg)
    output = Path(output_dir)
    where = resolve_device(device)
    validate_output_dir(output, format_name=TRAIN_FORMAT, overwrite=overwrite, resume=resume)
    if resume and not (output / "checkpoints/last.pt").is_file():
        raise FileNotFoundError("--resume requires output-dir/checkpoints/last.pt")
    previous = load_checkpoint(output / "checkpoints/last.pt") if resume else None
    if previous is not None:
        if previous["seed"] != seed or previous["resume_contract"] != resume_contract(cfg):
            raise ValueError("Resume seed or trained configuration differs from the last checkpoint")
        if cfg["training"]["epochs"] < previous["epoch"]:
            raise ValueError("Cannot resume to fewer epochs than the last checkpoint")
        if previous["best_epoch"] and not (output / "checkpoints/best.pt").is_file():
            raise FileNotFoundError("Resume requires the matching best.pt as well as last.pt")
    if overwrite and output.exists():
        shutil.rmtree(output)  # validate_output_dir has verified our format marker
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {"format": TRAIN_FORMAT, "model_id": "option_c", "seed": int(seed), "status": "preparing",
                "created_at": datetime.now(timezone.utc).isoformat(), "versions": versions(),
                "device": str(where), "config": cfg, "implementation": source_fingerprint(),
                "contract": {"nodes": "learned_categorical_diffusion_with_training_empirical_marginal_noise",
                             "edges": "gaussian_diffusion_of_scaled_weighted_adjacency",
                             "edge_prediction": "clean_scalar_weight_no_edge_softmax",
                             "spectral_role": "differentiable_auxiliary_loss_on_predicted_weighted_adjacency",
                             "spectral_stochastic_state": False, "degree_vae_required": False,
                             "eigenbasis_required": False, "rewiring_during_training": False,
                             "test_used_for_training": False}}
    try:
        train_rows, val_rows, schema, provenance = load_data(cfg)
        if previous is not None and previous["dataset"]["preprocessing_cache_key"] != provenance["preprocessing_cache_key"]:
            raise ValueError("Prepared data or preprocessing changed since this checkpoint; refusing resume")
        manifest.update(dataset=provenance, status="training")
        write_json(output / "manifest.json", manifest)
        atomic_write(output / "resolved_config.yaml", lambda p: p.write_text(yaml.safe_dump({"option_c": cfg}, sort_keys=False)))
        write_json(output / "schema.json", schema)
        seed_everything(seed)
        basis = TypedGraphletsMulti.from_schema(schema)
        mc = model_config(cfg, schema, basis)
        model = OptionCDenoiser(**mc).to(where)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        schedule = cosine_alpha_bar(cfg["diffusion"]["steps"], device=where)
        node_noise = MarginalNoise(schema["node_marginal"], schedule)
        corrupt_rng = torch.Generator(device=where).manual_seed(seed+91)
        order_rng = np.random.default_rng(seed+92)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg["training"]["ema_decay"] > 0 else None
        eval_model = copy.deepcopy(model) if ema is not None else model
        history, first_epoch, optimizer_steps = [], 1, 0
        best, best_epoch = float("inf"), 0
        if previous is not None:
            model.load_state_dict(previous["model_state"])
            optimizer.load_state_dict(previous["optimizer_state"])
            history, optimizer_steps = previous["history"], previous["optimizer_steps"]
            first_epoch = previous["epoch"]+1
            best, best_epoch = previous["best_val_loss"], previous["best_epoch"]
            corrupt_rng.set_state(previous["rng"]["corruption"])
            order_rng.bit_generator.state = previous["rng"]["order"]
            torch.set_rng_state(previous["rng"]["torch_cpu"])
            if where.type == "cuda" and previous["rng"]["torch_cuda"]:
                torch.cuda.set_rng_state_all(previous["rng"]["torch_cuda"])
            if ema is not None:
                ema = {k: v.to(where) for k, v in previous["ema_state"].items()}
        manifest["parameter_count"] = sum(p.numel() for p in model.parameters())
        manifest["checkpoint_selection_weights"] = "ema" if ema is not None else "raw"
        write_json(output / "manifest.json", manifest)
        # Resume removes only log entries after the last successfully saved epoch.
        (output / "history.jsonl").write_text("".join(json.dumps(r)+"\n" for r in history))
        def checkpoint(epoch, *, resumable):
            out = {"format": CHECKPOINT_FORMAT, "config": cfg, "generation_contract": generation_contract(cfg),
                   "resume_contract": resume_contract(cfg), "seed": int(seed), "epoch": epoch,
                   "optimizer_steps": optimizer_steps, "best_epoch": best_epoch, "best_val_loss": best,
                   "model_config": mc, "model_state": cpu_tree(model.state_dict()),
                   "ema_state": cpu_tree(ema), "schema": schema, "dataset": provenance,
                   "versions": versions(), "device": str(where), "implementation": manifest["implementation"]}
            if resumable:
                out.update(optimizer_state=cpu_tree(optimizer.state_dict()), history=history,
                           rng={"corruption": corrupt_rng.get_state().cpu(), "order": order_rng.bit_generator.state,
                                "torch_cpu": torch.get_rng_state(),
                                "torch_cuda": torch.cuda.get_rng_state_all() if where.type == "cuda" else []})
            return out
        tc = cfg["training"]
        with (output / "history.jsonl").open("a") as log:
            for epoch in range(first_epoch, tc["epochs"]+1):
                epoch_started = time.monotonic()
                metrics, steps = _run_epoch(model, train_rows, basis, cfg, schedule, node_noise, corrupt_rng, where,
                                            order=order_rng.permutation(len(train_rows)), optimizer=optimizer,
                                            ema_state=ema, epoch=epoch)
                optimizer_steps += steps
                row = {"epoch": epoch, "optimizer_steps": optimizer_steps, **{"train_"+k: v for k, v in metrics.items()}}
                validate = epoch == 1 or epoch % tc["validation_every"] == 0 or epoch == tc["epochs"]
                if validate:
                    if ema is not None:
                        eval_model.load_state_dict(ema)
                    vrng = torch.Generator(device=where).manual_seed(tc["validation_seed"])
                    vm, _ = _run_epoch(eval_model, val_rows, basis, cfg, schedule, node_noise, vrng, where,
                                       order=np.arange(len(val_rows)), epoch=epoch)
                    row.update({"val_"+k: v for k, v in vm.items()})
                    if vm["loss"] < best:
                        best, best_epoch = vm["loss"], epoch
                        save_torch(output / "checkpoints/best.pt", checkpoint(epoch, resumable=False))
                row["epoch_seconds"] = time.monotonic()-epoch_started
                history.append(row)
                log.write(json.dumps(row, allow_nan=False)+"\n")
                log.flush()
                if epoch == 1 or validate or epoch % tc["checkpoint_every"] == 0 or epoch == tc["epochs"]:
                    save_torch(output / "checkpoints/last.pt", checkpoint(epoch, resumable=True))
                    write_json(output / "training_metrics.json", {"history": history, "best_epoch": best_epoch, "best_val_loss": best})
                    manifest.update(last_saved_epoch=epoch, optimizer_steps=optimizer_steps,
                                    best_epoch=best_epoch, best_val_loss=best)
                    write_json(output / "manifest.json", manifest)
                if epoch == 1 or epoch % tc["log_every"] == 0 or epoch == tc["epochs"]:
                    print(f"[option-c] epoch={epoch}/{tc['epochs']} train_loss={row['train_loss']:.6f} "
                          f"best_val={best:.6f} best_epoch={best_epoch}", flush=True)
        manifest.update(status="completed", epochs_completed=tc["epochs"], optimizer_steps=optimizer_steps,
                        best_epoch=best_epoch, best_val_loss=best, duration_seconds_this_invocation=time.monotonic()-started,
                        checkpoint={"best": "checkpoints/best.pt", "last": "checkpoints/last.pt"})
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        # Keep recoverable checkpoints. Never report an interrupted run as completed.
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "manifest.json", manifest)
        raise
