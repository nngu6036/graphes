#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from grapher.construction.coarse import assert_constructor_validity, construct_coarse_graph
from grapher.data.io import load_dataset_splits
from grapher.properties.sampler import LearnedSummarySampler
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.refinement.rewiring import apply_action
from grapher.utils.io import load_yaml


def _load_graphs(config: dict[str, Any]):
    dcfg = config.get("dataset", {}) or {}
    splits = load_dataset_splits(
        dcfg.get("name", "sbm_spectre"),
        root=dcfg.get("root", "outputs/datasets"),
        build_if_missing=bool(dcfg.get("build_if_missing", True)),
        config_path=dcfg.get("config_path"),
    )
    return splits


def _find_metrics_file(output_dir: Path) -> Path:
    candidates = list(output_dir.rglob("*metrics*.json"))
    if not candidates:
        candidates = list(output_dir.rglob("*.json"))
    for path in candidates:
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if "metrics" in obj or any(k in obj for k in ["coarse", "grapher_opt", "learned_selector"]):
            return path
    raise FileNotFoundError(f"No metrics JSON found under {output_dir}")


def _metric_block(obj: dict[str, Any]) -> dict[str, Any]:
    return obj.get("metrics", obj)


def _get_method(metrics: dict[str, Any], *names: str) -> dict[str, float]:
    for name in names:
        if name in metrics:
            return metrics[name]
    raise KeyError(f"Missing method among {names}. Available: {list(metrics)}")


def _get(m: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in m:
            val = m[name]
            if val is None:
                return float("nan")
            return float(val)
    raise KeyError(f"Missing metric among {names}. Available: {list(m)}")


def check_dataset(config: dict[str, Any]) -> None:
    splits = _load_graphs(config)

    total = 0
    errors = []

    for split, graphs in splits.items():
        for i, g in enumerate(graphs):
            total += 1
            if not isinstance(g, nx.Graph):
                errors.append(f"{split}[{i}] is not nx.Graph")
                continue
            if g.is_directed():
                errors.append(f"{split}[{i}] is directed")
            if nx.number_of_selfloops(g) > 0:
                errors.append(f"{split}[{i}] has self-loops")
            if g.number_of_nodes() > 1 and not nx.is_connected(g):
                errors.append(f"{split}[{i}] is disconnected")

    if errors:
        preview = "\n".join(errors[:10])
        raise AssertionError(f"Dataset check failed with {len(errors)} errors:\n{preview}")

    print(f"PASS dataset check total_graphs={total}")


def check_summary_generator(config: dict[str, Any]) -> None:
    splits = _load_graphs(config)
    train_graphs = list(splits["train"])

    scfg = SummaryConfig.from_dict(config.get("summary", {}) or {}, train_graphs)
    sampler = LearnedSummarySampler.from_config(
        config["summary_generator"],
        seed=int(config.get("seed", 0)),
    )

    rng = np.random.default_rng(int(config.get("seed", 0)))
    constructor_cfg = config.get("constructor", {}) or {}
    require_connected = bool(constructor_cfg.get("ensure_connected", True))

    graphical = []
    valid = []

    for _ in range(100):
        summary = sampler.sample(rng)
        degree_sequence = [int(d) for d in summary["degree_sequence"]]
        graphical.append(nx.is_graphical(degree_sequence, method="eg"))

        try:
            g = construct_coarse_graph(summary, constructor_cfg, rng)
            assert_constructor_validity(g, summary, require_connected=require_connected)
            valid.append(True)
        except Exception:
            valid.append(False)

    graphical_rate = float(np.mean(graphical))
    valid_rate = float(np.mean(valid))

    if graphical_rate < 0.95 or valid_rate < 0.95:
        raise AssertionError(
            f"Summary generator failed: graphical_rate={graphical_rate:.3f}, "
            f"valid_rate={valid_rate:.3f}"
        )

    print(
        f"PASS summary_generator graphical_rate={graphical_rate:.3f} "
        f"valid_rate={valid_rate:.3f}"
    )


def _iter_teacher_records(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_teacher(config: dict[str, Any]) -> None:
    tcfg = config.get("teacher", {}) or {}
    teacher_dir = Path(tcfg.get("output_dir", "outputs/teachers/sbm_report"))
    train_path = teacher_dir / "train.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing teacher cache: {train_path}")

    n_records = 0
    positive_delta = 0
    entropies = []
    max_probs = []

    for rec in _iter_teacher_records(train_path):
        n_records += 1

        actions = rec.get("actions", [])
        probs = np.asarray(rec.get("teacher_probs", []), dtype=float)
        deltas = np.asarray([a.get("delta_energy", 0.0) for a in actions], dtype=float)

        if len(actions) == 0:
            raise AssertionError(f"Record {n_records} has no actions")
        if probs.shape[0] != len(actions):
            raise AssertionError(f"Record {n_records} probability/action mismatch")
        if not np.isclose(probs.sum(), 1.0, atol=1e-4):
            raise AssertionError(f"Record {n_records} probabilities do not sum to 1")

        positive_delta += float(np.max(deltas) > 0.0)

        eps = 1e-12
        ent = float(-(probs * np.log(probs + eps)).sum())
        entropies.append(ent)
        max_probs.append(float(probs.max()))

    if n_records < 1000:
        raise AssertionError(f"Too few teacher records: {n_records}")

    positive_rate = positive_delta / max(n_records, 1)
    mean_entropy = float(np.mean(entropies))
    mean_max_prob = float(np.mean(max_probs))

    if positive_rate < 0.5:
        raise AssertionError(f"Teacher positive_delta_rate too low: {positive_rate:.3f}")

    print(
        f"PASS teacher n_records={n_records} "
        f"positive_delta_rate={positive_rate:.3f} "
        f"mean_entropy={mean_entropy:.3f} "
        f"mean_max_prob={mean_max_prob:.3f}"
    )


def check_selector(config: dict[str, Any]) -> None:
    scfg = config.get("selector", {}) or {}
    ckpt = Path(scfg.get("checkpoint_path", "outputs/selectors/sbm_report/checkpoint.pt"))

    if not ckpt.exists():
        raise FileNotFoundError(f"Missing selector checkpoint: {ckpt}")

    report_candidates = [
        ckpt.parent / "training_report.json",
        ckpt.parent / "metrics.json",
        ckpt.parent / "selector_report.json",
    ]

    report_path = None
    for p in report_candidates:
        if p.exists():
            report_path = p
            break

    if report_path is None:
        print(f"PASS selector checkpoint exists: {ckpt}")
        print("WARN no selector training report found; skipping loss/top-k checks")
        return

    report = json.loads(report_path.read_text())

    val_loss = report.get("val_loss", report.get("best_val_loss"))
    top1 = report.get("top1", report.get("val_top1"))
    mean_delta = report.get("mean_predicted_delta", report.get("val_mean_predicted_delta"))

    if val_loss is not None and not math.isfinite(float(val_loss)):
        raise AssertionError("Selector val_loss is not finite")

    if top1 is not None and float(top1) <= 0.0:
        raise AssertionError(f"Selector top1 is invalid: {top1}")

    if mean_delta is not None and float(mean_delta) <= 0.0:
        raise AssertionError(f"Selector mean predicted delta <= 0: {mean_delta}")

    print(
        f"PASS selector checkpoint={ckpt} "
        f"val_loss={val_loss} top1={top1} mean_delta={mean_delta}"
    )


def check_generation(config: dict[str, Any], output_dir: Path) -> None:
    metrics_path = _find_metrics_file(output_dir)
    obj = json.loads(metrics_path.read_text())
    metrics = _metric_block(obj)

    coarse = _get_method(metrics, "coarse")
    random = _get_method(metrics, "random_rewire", "random")
    learned = _get_method(metrics, "learned_selector", "grapher_selector", "selector", "grapher_opt")

    # Hard constraints.
    deg_coarse = _get(coarse, "degree", "degree_mmd")
    deg_learned = _get(learned, "degree", "degree_mmd")
    conn = _get(learned, "conn", "connectedness", "connectedness_rate")

    if abs(deg_learned - deg_coarse) > 1e-8:
        raise AssertionError(
            f"Degree MMD changed unexpectedly: coarse={deg_coarse}, learned={deg_learned}"
        )

    if conn < 0.999:
        raise AssertionError(f"Connectedness too low: {conn}")

    # Improvement against coarse.
    for name, aliases in {
        "clustering": ("clustering", "clustering_mmd"),
        "spectral": ("spectral", "spectral_mmd"),
        "motif": ("motif", "motif_proxy_mmd"),
    }.items():
        c = _get(coarse, *aliases)
        l = _get(learned, *aliases)
        if not (l < c):
            raise AssertionError(f"{name} did not improve over coarse: coarse={c}, learned={l}")

    # Prefer learned selector to beat random on at least 2 of 3 cheap non-degree metrics.
    wins = 0
    for name, aliases in {
        "clustering": ("clustering", "clustering_mmd"),
        "spectral": ("spectral", "spectral_mmd"),
        "motif": ("motif", "motif_proxy_mmd"),
    }.items():
        l = _get(learned, *aliases)
        r = _get(random, *aliases)
        wins += int(l < r)

    if wins < 2:
        raise AssertionError(f"Learned selector wins only {wins}/3 metrics against random")

    # Orbit is optional but checked when present and finite.
    try:
        orbit_l = _get(learned, "orbit", "orbit_mmd")
        orbit_c = _get(coarse, "orbit", "orbit_mmd")
        if math.isfinite(orbit_l) and math.isfinite(orbit_c) and not (orbit_l < orbit_c):
            raise AssertionError(f"Orbit did not improve: coarse={orbit_c}, learned={orbit_l}")
    except KeyError:
        pass

    print(f"PASS generation metrics from {metrics_path}")
    print(f"learned_vs_random_wins={wins}/3 cheap metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--check",
        choices=[
            "dataset",
            "summary_generator",
            "teacher",
            "selector",
            "generation",
            "all",
        ],
        required=True,
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.check in {"dataset", "all"}:
        check_dataset(config)

    if args.check in {"summary_generator", "all"}:
        check_summary_generator(config)

    if args.check in {"teacher", "all"}:
        check_teacher(config)

    if args.check in {"selector", "all"}:
        check_selector(config)

    if args.check in {"generation", "all"}:
        if args.output_dir is None:
            raise ValueError("--output-dir is required for generation check")
        check_generation(config, Path(args.output_dir))


if __name__ == "__main__":
    main()