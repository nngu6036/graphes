#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from grapher.generators.summary_vae import (
    ConditionalSummaryVAE,
    load_summary_vae_checkpoint,
)
from grapher.properties.summary import summary_to_jsonable
from grapher.utils.device import resolve_torch_device
from grapher.utils.io import ensure_dir, load_json, save_json, save_pickle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample structural summaries from a trained SummaryVAE checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/summary_generators/samples")
    parser.add_argument(
        "--device", default="auto", help="Torch device. CUDA is required."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax/rounded postprocessing rather than categorical sampling.",
    )
    parser.add_argument(
        "--condition-json",
        default=None,
        help=(
            "JSON degree-summary object required by conditional checkpoints. "
            "The same condition is reused for all requested samples."
        ),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = resolve_torch_device(args.device)
    out_dir = ensure_dir(args.output_dir)
    print(f"Using device: {device}", flush=True)
    model, vectorizer, _ = load_summary_vae_checkpoint(args.checkpoint, device=device)
    condition_summaries = None
    with torch.no_grad():
        if isinstance(model, ConditionalSummaryVAE):
            if not args.condition_json:
                raise ValueError(
                    "--condition-json is required for a conditional "
                    "target-summary checkpoint."
                )
            condition_summary = load_json(args.condition_json)
            condition_summaries = [condition_summary] * int(args.num_samples)
            condition_np = np.repeat(
                vectorizer.to_condition_vector(condition_summary)[None, :],
                int(args.num_samples),
                axis=0,
            )
            condition = torch.as_tensor(
                condition_np,
                dtype=torch.float32,
                device=device,
            )
            outputs = model.sample_outputs(condition, device=device)
        else:
            outputs = model.sample_outputs(int(args.num_samples), device=device)
    summaries = vectorizer.outputs_to_summaries(
        outputs,
        rng=rng,
        deterministic=bool(args.deterministic),
        condition_summaries=condition_summaries,
    )
    save_pickle(summaries, Path(out_dir) / "summaries.pkl")
    save_json(
        [summary_to_jsonable(s) for s in summaries], Path(out_dir) / "summaries.json"
    )
    print(f"Saved {len(summaries)} summaries to: {out_dir}")


if __name__ == "__main__":
    main()
