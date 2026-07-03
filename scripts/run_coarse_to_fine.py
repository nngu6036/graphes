#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from grapher.pipeline.coarse_to_fine import run_coarse_to_fine
from grapher.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fresh coarse-to-fine graph generation pipeline.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to outputs/<experiment>.")
    parser.add_argument("--num-generate", type=int, default=None, help="Override evaluation.num_generate.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = args.output_dir or str(Path("outputs") / "coarse_to_fine" / str(config.get("experiment", "run")))
    result = run_coarse_to_fine(config, output_dir=output_dir, num_generate=args.num_generate)

    print("\nMetric summary")
    for method, metrics in result["metrics"].items():
        print(f"{method:>16s} | degree={metrics['degree_mmd']:.6f} | clustering={metrics['clustering_mmd']:.6f} | spectral={metrics['spectral_mmd']:.6f} | motif={metrics['motif_proxy_mmd']:.6f} | orbit={metrics['orbit_mmd']:.6f} | conn={metrics['connectedness_rate']:.3f}")
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
