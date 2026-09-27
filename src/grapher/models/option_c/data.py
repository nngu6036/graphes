"""Frozen prepared splits, train-only categorical priors, and cached graphlets.

The cache is shared across training seeds. It is keyed by source split hashes,
representation/target configuration and preprocessing implementation. Test data
is neither loaded nor used to fit a marginal, vocabulary, or checkpoint.
"""
from __future__ import annotations

import inspect
import json
import pickle
import warnings
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import yaml

from grapher.models.gdsm_simple.categorical.data import encode_graph, decode_graph, topology_summary
from grapher.models.gdsm_simple.categorical.multiscale import (
    TypedGraphletsMulti, count_multi, fit_basis, pack_training_counts,
    remap_training_counts, pack_basis_counts,
)
from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle
from .diffusion import WeightedEdgeCodec
from .runtime import sha256, object_hash, atomic_write, write_json

CACHE_FORMAT = "grapher_option_c_preprocessed_v1"


def _protocol_id(config):
    return config.get("protocol_id") or (config.get("protocol") or {}).get("id")


def dataset_provenance(cfg: dict) -> dict:
    dc = cfg["dataset"]
    path = Path(dc["root"]) / dc["name"]
    config_path = Path(dc["config_path"])
    if not config_path.is_file():
        raise FileNotFoundError(f"Frozen dataset config is missing: {config_path}")
    source = yaml.safe_load(config_path.read_text())
    if not isinstance(source, dict) or source.get("name") != dc["name"]:
        raise ValueError("The dataset config's serialized name differs from option_c.dataset.name")
    hashes = {}
    for split in ("train", "val"):
        split_path = path / f"{split}.pkl"
        if not split_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared split {split_path}. Use the existing project preparation command; "
                "Option C never rebuilds, re-splits, filters, or downloads a dataset.")
        hashes[split] = sha256(split_path)
    prepared = path / "resolved_dataset_config.yaml"
    prepared_hash = None
    if prepared.is_file():
        stored = yaml.safe_load(prepared.read_text()) or {}
        expected, actual = _protocol_id(source), _protocol_id(stored)
        if expected is not None and actual is not None and expected != actual:
            raise ValueError(f"Prepared dataset protocol {actual!r} differs from configured {expected!r}")
        prepared_hash = sha256(prepared)
    else:
        warnings.warn("Prepared split config is absent; split hashes are recorded but protocol identity cannot be independently verified.")
    return {"benchmark_id": dc["benchmark"], "serialized_id": dc["name"], "root": dc["root"],
            "config_path": str(config_path), "config_sha256": sha256(config_path),
            "prepared_config_sha256": prepared_hash, "protocol_id": _protocol_id(source),
            "split_sha256": hashes, "test_used_for_training": False,
            "dataset_config_snapshot": source}


def _load_graphs(path: Path) -> list[nx.Graph]:
    # Prepared project pickles are trusted local artifacts, not a safe untrusted format.
    graphs = load_trusted_networkx_pickle(path)
    if not isinstance(graphs, (list, tuple)) or not graphs:
        raise ValueError(f"Expected a nonempty prepared graph sequence at {path}")
    return list(graphs)


def _check_split_sizes(provenance, train, val):
    source = provenance["dataset_config_snapshot"]
    split = source.get("split", {})
    for name, graphs in (("train", train), ("val", val)):
        expected = split.get(name)
        if isinstance(expected, float) and source.get("num_graphs") is not None:
            expected = int(round(expected * int(source["num_graphs"])))
        if name == "val" and split.get("validation_count") is not None:
            expected = int(split["validation_count"])
        if isinstance(expected, int) and expected != len(graphs):
            raise ValueError(f"Frozen {name} split expects {expected} graphs; found {len(graphs)}. No subsampling is allowed.")


def _record(graph, vocab, codec, cfg):
    x, e = encode_graph(graph, vocab, cfg["model"]["max_nodes"])
    w = codec.encode(e)
    clustering, orbit = topology_summary(e, cfg["graphlets"]["clustering_bins"])
    counts = count_multi(x, e, cfg["graphlets"]["sizes"], limit=cfg["graphlets"]["max_connected_subsets"])
    return {"x": x, "e": e, "w": w,
            "spectrum": (np.linalg.eigvalsh(w.astype(np.float64)) / len(x)**.5).astype(np.float32),
            "clustering": clustering, "orbit": orbit, "counts": counts}


def prepare_records(train_graphs, val_graphs, cfg):
    """No eigenvectors or degree anchors are constructed, cached, or sampled."""
    if not train_graphs or not val_graphs:
        raise ValueError("Nonempty disjoint prepared train and validation splits are required")
    vocab = GraphCategoryVocabulary.from_graphs(train_graphs, cfg["categories"])
    codec = WeightedEdgeCodec(vocab, cfg["edge_representation"])
    orders = cfg["graphlets"]["sizes"]
    aggregate = {k: Counter() for k in orders}
    codebooks = {k: {} for k in orders}
    nodes = np.zeros(vocab.num_node_categories, np.int64)
    sizes = Counter()
    train, val = [], []
    for i, graph in enumerate(train_graphs):
        row = _record(graph, vocab, codec, cfg)
        for k in orders:
            aggregate[k].update(row["counts"][k])
        row["counts"] = pack_training_counts(row["counts"], codebooks)
        nodes += np.bincount(row["x"], minlength=len(nodes))
        sizes[len(row["x"])] += 1
        train.append(row)
        if (i + 1) % 10000 == 0:
            print(f"[option-c] prepared {i+1}/{len(train_graphs)} training graphs", flush=True)
    basis, coverage = fit_basis(aggregate, cfg["graphlets"])
    lookup = {k: np.array([basis.index[k].get(key, len(basis.keys_by_order[k])) for key in codebooks[k]], np.int32)
              for k in orders}
    for row in train:
        row["counts"] = remap_training_counts(row["counts"], lookup)
    del aggregate, codebooks, lookup
    for i, graph in enumerate(val_graphs):
        row = _record(graph, vocab, codec, cfg)
        row["counts"] = pack_basis_counts(row["counts"], basis)
        val.append(row)
        if (i + 1) % 10000 == 0:
            print(f"[option-c] prepared {i+1}/{len(val_graphs)} validation graphs", flush=True)
    pseudo = cfg["node_noise"]["pseudocount"]
    marginal = (nodes + pseudo) / (nodes.sum() + pseudo * len(nodes))
    schema = {"category_vocabulary": vocab.to_dict(), "node_marginal": marginal.tolist(),
              "node_category_counts": nodes.tolist(), "graph_sizes": sorted(sizes),
              "graph_size_counts": [int(sizes[n]) for n in sorted(sizes)],
              "size_prior_source": "training_split_only", "node_prior_source": "training_split_only",
              "node_distribution_role": "terminal_categorical_noise_not_fixed_final_composition",
              "edge_physical_weights_by_index": codec.values.tolist(), "edge_scale": codec.scale,
              "edge_thresholds_physical": codec.thresholds.tolist(),
              "training_graphlet_vocabulary_coverage": coverage, **basis.schema()}
    return train, val, schema


def load_data(cfg: dict):
    provenance = dataset_provenance(cfg)
    code_paths = {Path(__file__), Path(inspect.getfile(count_multi)), Path(inspect.getfile(encode_graph)),
                  Path(inspect.getfile(GraphCategoryVocabulary)), Path(inspect.getfile(WeightedEdgeCodec))}
    key = object_hash({"format": CACHE_FORMAT, "provenance": provenance,
                       "categories": cfg["categories"], "edge_representation": cfg["edge_representation"],
                       "graphlets": cfg["graphlets"], "node_noise": cfg["node_noise"],
                       "max_nodes": cfg["model"]["max_nodes"],
                       "preprocessing_code": {str(p.resolve().relative_to(Path(__file__).resolve().parents[2])): sha256(p)
                                              for p in sorted(code_paths)}})
    cache = Path(cfg["training"]["cache_dir"]) / cfg["dataset"]["benchmark"] / key
    cache_file = cache / "records.pkl"
    if cache_file.is_file():
        with cache_file.open("rb") as handle:
            data = pickle.load(handle)
        if data.get("format") != CACHE_FORMAT or data.get("cache_key") != key:
            raise ValueError("Invalid Option-C data cache; remove this cache directory explicitly")
        print(f"[option-c] reused preprocessing cache {cache}", flush=True)
    else:
        directory = Path(cfg["dataset"]["root"]) / cfg["dataset"]["name"]
        train = _load_graphs(directory / "train.pkl")
        val = _load_graphs(directory / "val.pkl")
        _check_split_sizes(provenance, train, val)
        print(f"[option-c] preparing ALL {len(train)} training + {len(val)} validation graphs; test is not loaded", flush=True)
        train, val, schema = prepare_records(train, val, cfg)
        data = {"format": CACHE_FORMAT, "cache_key": key, "train": train, "val": val, "schema": schema}
        def save(tmp):
            with tmp.open("wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write(cache_file, save)
        write_json(cache / "schema.json", schema)
        write_json(cache / "provenance.json", provenance)
    provenance.update(num_train_graphs_used=len(data["train"]), num_val_graphs_used=len(data["val"]),
                      preprocessing_cache_key=key, preprocessing_cache_path=str(cache))
    return data["train"], data["val"], data["schema"], provenance


def collate(records, basis: TypedGraphletsMulti, *, device):
    b, nmax = len(records), max(len(row["x"]) for row in records)
    x = np.zeros((b, nmax), np.int64)
    w = np.zeros((b, nmax, nmax), np.float32)
    spectrum = np.zeros((b, nmax), np.float32)
    mask = np.zeros((b, nmax), bool)
    hist, mass = [], []
    for i, row in enumerate(records):
        n = len(row["x"])
        x[i, :n], w[i, :n, :n], spectrum[i, :n], mask[i, :n] = row["x"], row["w"], row["spectrum"], True
        h, m = basis.encode_counts(row["counts"], n)
        hist.append(h)
        mass.append(m)
    out = {"x": x, "w": w, "spectrum": spectrum, "mask": mask, "histogram": np.stack(hist),
           "mass": np.stack(mass), "graphlet_order_mask": np.array([[len(r["x"]) >= k for k in basis.orders] for r in records]),
           "clustering": np.stack([r["clustering"] for r in records]), "orbit": np.stack([r["orbit"] for r in records])}
    return {key: torch.as_tensor(value, device=device) for key, value in out.items()}


def permute_aligned(batch, generator):
    x, w = batch["x"].clone(), batch["w"].clone()
    for i, n in enumerate(batch["mask"].sum(1).tolist()):
        p = torch.randperm(n, device=x.device, generator=generator)
        x[i, :n] = x[i, :n][p]
        w[i, :n, :n] = w[i, :n, :n][p][:, p]
    # Ordered eigenvalues and graph-level targets are NOT node-indexed.
    return {**batch, "x": x, "w": w}
