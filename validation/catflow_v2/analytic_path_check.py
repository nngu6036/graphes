#!/usr/bin/env python3
"""One binary undirected edge: exact posterior, not a trained graph benchmark.

The category-coordinate difference has mean t*(2*y-1) and variance
(1-t)**2 + extra_noise_std**2. The symmetrized per-category Gaussian edge
prior makes its difference variance one. A fixed 0.5 coordinate perturbation
has difference variance 0.25. We integrate the SAME straight-line velocity
with the exact posterior for each training law, starting from N(0,1).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit


def run(num_samples=200000, steps=400, seed=1729):
    initial = np.random.default_rng(seed).standard_normal(num_samples)
    rows = []
    endpoint = 0.95
    for probability in (0.2, 0.35, 0.5):
        for extra in (0.0, 0.5):
            state = initial.copy()
            dt = endpoint / steps
            def velocity(value, time):
                variance = (1 - time)**2 + extra**2
                posterior = expit(np.log(probability / (1 - probability)) + 2*time*value/variance)
                return (2*posterior - 1 - value) / (1 - time)
            for step in range(steps):
                t = step * dt
                a = velocity(state, t)
                b = velocity(state + dt*a/2, t + dt/2)
                c = velocity(state + dt*b/2, t + dt/2)
                d = velocity(state + dt*c, t + dt)
                state += dt*(a + 2*b + 2*c + d)/6
            rows.append(dict(target_edge_probability=probability, training_extra_noise_std=extra,
                generated_edge_frequency=float((state > 0).mean()), num_samples=num_samples,
                endpoint=endpoint, method="rk4", steps=steps))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=200000)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.num_samples, args.steps) < 1:
        parser.error("Samples and integration steps must be positive.")
    text = json.dumps(run(args.num_samples, args.steps, args.seed), indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")

if __name__ == "__main__":
    main()
