#!/usr/bin/env python3
"""Train Option C from frozen prepared train/val splits (not a baseline wrapper)."""
from __future__ import annotations

import argparse
from pathlib import Path

from grapher.models.option_c.config import load_config, validate_config
from grapher.models.option_c.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Standalone option_c YAML")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto", help="cpu, gpu, cuda[:index], or auto; explicit gpu never falls back")
    parser.add_argument("--epochs", type=int, default=None, help="Explicit override, recorded in resolved_config.yaml")
    parser.add_argument("--batch-size", type=int, default=None, help="Explicit training batch-size override")
    parser.add_argument("--cpu-threads", type=int, default=None, help="Optional explicit torch CPU thread limit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume output-dir/checkpoints/last.pt including optimizer/RNG state")
    mode.add_argument("--overwrite", action="store_true", help="Replace an output directory only if it belongs to Option C")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        import torch
        torch.set_num_threads(args.cpu_threads)
    cfg = validate_config(cfg)
    result = train(cfg, args.output_dir, seed=args.seed, device=args.device, resume=args.resume, overwrite=args.overwrite)
    print(f"[option-c] best checkpoint: {args.output_dir / 'checkpoints/best.pt'} "
          f"(epoch={result['best_epoch']}, val_loss={result['best_val_loss']:.6f})", flush=True)


if __name__ == "__main__":
    main()
