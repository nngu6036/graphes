#!/usr/bin/env python3
"""Audit a trusted local CatFlow run without retraining or modifying artifacts.

Run with the model interpreter and PYTHONPATH=src. Checkpoints and graph pickles
must be your own trusted artifacts; PyTorch/pickle loading is not sandboxed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

WORKERS = Path(__file__).resolve().parents[1] / "src/grapher/models/external_workers"
sys.path.insert(0, str(WORKERS))
from common import selective_module, torch_load
from catflow_path import probe_upstream_path
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_statistics(graphs):
    graphs = list(graphs)
    if not graphs:
        return {"count": 0}
    if any(not isinstance(g, nx.Graph) or g.is_directed() for g in graphs):
        raise TypeError("Expected a list of undirected NetworkX graphs.")
    n = np.asarray([len(g) for g in graphs])
    m = np.asarray([g.number_of_edges() for g in graphs])
    return {"count": len(graphs), "mean_nodes": float(n.mean()), "mean_edges": float(m.mean()),
            "node_count_min": int(n.min()), "node_count_max": int(n.max()),
            "mean_degree_per_graph": float(np.mean([2 * e / v if v else 0 for v, e in zip(n, m)])),
            "mean_density": float(np.mean([nx.density(g) for g in graphs])),
            "mean_clustering": float(np.mean([nx.average_clustering(g) if len(g) else 0 for g in graphs])),
            "mean_triangles": float(np.mean([sum(nx.triangles(g).values()) / 3 for g in graphs])),
            "connected_fraction": float(np.mean([nx.is_connected(g) if len(g) else False for g in graphs])),
            "empty_edge_fraction": float(np.mean(m == 0)),
            "isolated_node_fraction": float(sum(sum(d == 0 for _, d in g.degree()) for g in graphs) / max(1, n.sum()))}


def summarize_npz(path: Path):
    with np.load(path, allow_pickle=False) as f:
        sizes, adjacency = f["num_nodes"], f["adjacency"]
        graphs = [nx.from_numpy_array(a[:int(n), :int(n)] != 0) for a, n in zip(adjacency, sizes)]
    return graph_statistics(graphs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/baselines/catflow/community_small/seed_42"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest_path = args.run_dir / "train/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    root = args.source_root or os.environ.get("CATFLOW") or (manifest or {}).get("runtime", {}).get("source_root")
    if not root:
        parser.error("Provide --source-root, set CATFLOW, or specify an existing run manifest.")
    source = Path(root).expanduser().resolve()
    source_file = source / "flow_matching.py"
    flow = selective_module("_catflow_audit_flow", source_file,
                            selected={"conditional_velocity", "hyperplane_proj"}, globals_dict={"torch": torch})
    result = {"source_root": str(source), "source_flow_matching_sha256": digest(source_file),
              "upstream_path_probe": probe_upstream_path(flow), "warnings": []}
    if result["upstream_path_probe"]["node_residual_std_at_t1"] > 1e-4:
        result["warnings"].append("Upstream normal path has nonvanishing endpoint noise. This affects legacy/upstream training, not checkpoints explicitly trained with path=linear.")
    if manifest:
        options = manifest["wrapper_options"]
        train_options = options["train"]
        expected_source = manifest.get("source", {}).get("files", {}).get("flow_matching.py")
        result.update(run_dir=str(args.run_dir), dataset=manifest["dataset"],
                      recorded_training_options=train_options,
                      recorded_sample_options=options.get("sample", {}),
                      source_matches_training=(result["source_flow_matching_sha256"] == expected_source),
                      recorded_source_flow_matching_sha256=expected_source)
        if expected_source != result["source_flow_matching_sha256"]:
            result["warnings"].append("Local flow_matching.py does not match the training source; its probe does NOT identify the trained probability path.")
        checkpoint = args.run_dir / "train" / manifest["checkpoint"]["path"]
        if checkpoint.is_file():
            state = torch_load(checkpoint, "cpu")
            result["checkpoint"] = {"path": str(checkpoint), "sha256_matches_manifest": digest(checkpoint) == manifest["checkpoint"]["sha256"],
                "actual_saved_epoch": state.get("epoch"), "optimizer_steps": state.get("optimizer_steps"),
                "ema_backend": state.get("ema_backend"), "ema_num_updates": state.get("ema", {}).get("num_updates"), "probability_path": state.get("probability_path", {"name": "legacy_upstream_unrecorded"}),
                "has_best_snapshot": state.get("best") is not None,
                "best_epoch": (state.get("best") or {}).get("epoch"), "best_val_ema": (state.get("best") or {}).get("val_ema")}
            history = state.get("history", [])
            result["history_tail"] = history[-5:]
            valid_history = [row for row in history if isinstance(row.get("val"), (float, int))]
            if valid_history:
                result["minimum_logged_validation"] = min(valid_history, key=lambda r: r["val"])
                result["minimum_logged_validation_note"] = ("Fixed-noise EMA validation on the held-out validation split." if state.get("format_version") == 2 else
                    "Legacy losses use fresh random draws and raw weights, not comparable fixed-EMA validation.")
        else:
            result["warnings"].append("Managed checkpoint is missing: " + str(checkpoint))
        for split in ("train", "val"):
            path = args.run_dir / "train/native_dataset" / (split + ".npz")
            if path.is_file():
                result[split + "_graph_statistics"] = summarize_npz(path)
                if split == "train":
                    steps = int(np.ceil(result[split + "_graph_statistics"]["count"] / int(train_options["batch_size"])))
                    result["optimizer_steps_per_epoch"] = steps
                    result["configured_total_optimizer_steps"] = steps * int(train_options["epochs"])
    else:
        result["warnings"].append("No training manifest found; this is a source-only audit, not an audit of your trained weights.")
    if args.generated_dir is not None:
        path = args.generated_dir / "base_graphs.pkl"
        graphs = load_trusted_networkx_pickle(path)
        if isinstance(graphs, dict):
            graphs = graphs.get("graphs", graphs.get("generated_graphs", graphs))
        result["generated_graph_statistics"] = graph_statistics(graphs)
        result["generated_graphs_sha256"] = digest(path)
        generated_manifest_path = args.generated_dir / "manifest.json"
        if generated_manifest_path.is_file():
            gm = json.loads(generated_manifest_path.read_text())
            result["generation_provenance"] = {"generation_seed": gm.get("generation_seed"),
                "num_generated": gm.get("num_generated"), "checkpoint_sha256": gm.get("checkpoint_sha256"),
                "graphs_match_manifest": gm.get("base_graphs", {}).get("sha256") == result["generated_graphs_sha256"],
                "checkpoint_matches_training_manifest": (gm.get("checkpoint_sha256") == manifest["checkpoint"]["sha256"]) if manifest else None}
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
