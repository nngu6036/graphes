#!/usr/bin/env python3
"""Generate Option-C graphs; no current dataset or degree-prior checkpoint is needed."""
from __future__ import annotations

import argparse
from pathlib import Path

from grapher.models.option_c.config import load_config, validate_config
from grapher.models.option_c.sampling import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-generate", "--num-samples", dest="num_generate", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="auto", help="cpu, gpu, cuda[:index], or auto")
    parser.add_argument("--steps", type=int, default=None, help="Explicit reverse-step override (exact skip kernels)")
    parser.add_argument("--batch-size", type=int, default=None, help="Explicit generation batch-size override")
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--no-refine", action="store_true", help="Paired no-swaps control; neural samples are unchanged")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name in ("steps", "batch_size", "sampler"):
        value = getattr(args, name)
        if value is not None:
            cfg["sampling"][name] = value
    if args.no_refine:
        cfg["refinement"]["enabled"] = False
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        import torch
        torch.set_num_threads(args.cpu_threads)
    cfg = validate_config(cfg)
    result = generate(cfg, args.checkpoint, args.output_dir, num_graphs=args.num_generate,
                      seed=args.seed, device=args.device, overwrite=args.overwrite)
    print(f"[option-c] saved {result['num_generated']} graphs to {args.output_dir / 'base_graphs.pkl'}", flush=True)


if __name__ == "__main__":
    main()
