from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import hashlib
import json

import networkx as nx

from grapher.data.builders import build_splits_from_config
from grapher.utils.io import (
    ensure_dir,
    load_pickle,
    load_yaml,
    save_json,
    save_pickle,
    save_yaml,
)


def dataset_dir(dataset: str, root: str | Path = "outputs/datasets") -> Path:
    return Path(root) / dataset


def save_dataset_splits(
    dataset: str,
    splits: dict[str, list[nx.Graph]],
    config: dict[str, Any],
    root: str | Path,
) -> None:
    out = ensure_dir(dataset_dir(dataset, root))
    for split, graphs in splits.items():
        save_pickle(graphs, out / f"{split}.pkl")
    save_yaml(config, out / "resolved_dataset_config.yaml")
    save_json(
        {"dataset": dataset, "split_sizes": {k: len(v) for k, v in splits.items()}},
        out / "metadata.json",
    )



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_split_fingerprint(split_sha256: Mapping[str, str]) -> str:
    record = {
        split: {"path": f"{split}.pkl", "sha256": str(split_sha256[split])}
        for split in sorted(split_sha256)
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_config_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dataset payload from either a bare or wrapped YAML config."""

    nested = raw.get("dataset")
    if isinstance(nested, Mapping):
        # Some report-facing YAML files wrap the actual dataset under `dataset`.
        # Prefer that mapping only when it contains the dataset contract itself.
        if any(key in nested for key in ("name", "split", "num_graphs", "protocol_id")):
            return dict(nested)
    return dict(raw)


def _expected_split_sizes(config: Mapping[str, Any]) -> dict[str, int] | None:
    split = config.get("split")
    if not isinstance(split, Mapping):
        return None
    # Molecular protocols can name index-based partition rules rather than
    # fractions. Their declared rules are compared against resolved metadata.
    try:
        train_fraction = float(split.get("train", 0.8))
        val_fraction = float(split.get("val", 0.1))
        test_fraction = float(split.get("test", 1.0 - train_fraction - val_fraction))
    except (TypeError, ValueError):
        return None
    values = {"train": train_fraction, "val": val_fraction, "test": test_fraction}
    total = config.get("num_graphs")
    # QM9 and fixed-count molecular subsets declare absolute split sizes,
    # often without the generic builder's num_graphs field.
    if (
        total is None
        or split.get("strategy") == "fixed_counts"
        or any(v > 1 for v in values.values())
    ):
        if all(
            key in split and value >= 0 and value.is_integer()
            for key, value in values.items()
        ):
            counts = {key: int(value) for key, value in values.items()}
            if total is not None and sum(counts.values()) != int(total):
                raise ValueError(
                    "Configured train/val/test counts must sum to num_graphs."
                )
            return counts
        if total is None:
            return None
    total = int(total)
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1.0e-8:
        raise ValueError("Configured train/val/test fractions must sum to 1.0.")
    n_train = int(round(total * train_fraction))
    n_val = int(round(total * val_fraction))
    return {"train": n_train, "val": n_val, "test": total - n_train - n_val}


def _normalized_split_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve index-based rules and controls while normalizing numeric values."""
    split = config.get("split", {})
    contract = dict(split)
    contract["seed"] = int(split.get("seed", config.get("seed", 0)))
    for name, default in (("train", 0.8), ("val", 0.1), ("test", 0.1)):
        value = split.get(name, default)
        try:
            contract[name] = float(value)
        except (TypeError, ValueError):
            contract[name] = value
    return contract




def _source_index_contract(
    splits: Mapping[str, Sequence[nx.Graph]],
) -> dict[str, list[int]] | None:
    """Return per-split precomputed-source indices when every graph has one.

    Unlike pickle-byte hashes, source indices are stable across Python, NetworkX,
    and pickle protocol versions.  They are therefore suitable for freezing a
    split of a pinned benchmark artifact such as SPECTRE Community-small.
    """

    contract: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        graphs = list(splits.get(split, []))
        indices: list[int] = []
        for graph in graphs:
            value = graph.graph.get("source_index")
            if value is None:
                return None
            indices.append(int(value))
        contract[split] = indices
    return contract

def _ego_split_content_sha256(
    splits: Mapping[str, Sequence[nx.Graph]],
) -> dict[str, str]:
    """Hash ordered, integer-labelled ego topology and source-center identity.

    JSON avoids pickle implementation details and NetworkX cached properties.
    Node/edge insertion order is ignored; graph order within each split matters.
    """
    hashes = {}
    for split in ("train", "val", "test"):
        records = []
        for graph in splits.get(split, []):
            if graph.is_directed() or graph.is_multigraph():
                raise ValueError("Ego content hashes require simple undirected graphs.")
            if any(type(node) is not int for node in graph):
                raise ValueError("Ego content hashes require integer node labels.")
            records.append({
                "nodes": sorted(graph.nodes()),
                "edges": sorted(sorted(edge) for edge in graph.edges()),
                "center": graph.graph.get("center"),
                "radius": graph.graph.get("radius"),
            })
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
        hashes[split] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashes


def validate_prepared_dataset_contract(
    dataset: str,
    *,
    root: str | Path = "outputs/datasets",
    config_path: str | Path | None = None,
    splits: Mapping[str, Sequence[nx.Graph]] | None = None,
    mismatch_policy: str = "error",
) -> list[str]:
    """Verify that prepared split files match the frozen dataset contract.

    This check prevents a stale prepared dataset (for example 70/10/20) from
    being silently reused after the benchmark config has been frozen to another
    split (for example 64/16/20).  When `frozen_split` hashes are present in the
    dataset config, the exact split artifacts are verified as well.
    """

    policy = str(mismatch_policy).strip().lower()
    if policy not in {"error", "warn", "ignore"}:
        raise ValueError(f"Unknown dataset mismatch policy: {mismatch_policy!r}")
    if config_path is None:
        return []

    config_file = Path(config_path)
    if not config_file.is_file():
        return []
    requested = _dataset_config_payload(load_yaml(config_file))
    directory = dataset_dir(dataset, root)
    paths = {split: directory / f"{split}.pkl" for split in ("train", "val", "test")}
    if not all(path.is_file() for path in paths.values()):
        return []

    if splits is None:
        actual_sizes = {split: len(load_pickle(path)) for split, path in paths.items()}
    else:
        actual_sizes = {split: len(list(splits.get(split, []))) for split in paths}

    mismatches: list[str] = []
    expected_sizes = _expected_split_sizes(requested)
    if expected_sizes is not None and actual_sizes != expected_sizes:
        mismatches.append(
            "prepared split sizes differ from the dataset config: "
            f"expected train/val/test={expected_sizes['train']}/{expected_sizes['val']}/{expected_sizes['test']}, "
            f"found {actual_sizes['train']}/{actual_sizes['val']}/{actual_sizes['test']}"
        )

    resolved_path = directory / "resolved_dataset_config.yaml"
    if resolved_path.is_file():
        resolved = _dataset_config_payload(load_yaml(resolved_path))
        requested_split = requested.get("split")
        resolved_split = resolved.get("split")
        if isinstance(requested_split, Mapping) and isinstance(resolved_split, Mapping):
            normalized_requested = _normalized_split_contract(requested)
            normalized_resolved = _normalized_split_contract(resolved)
            if normalized_requested != normalized_resolved:
                mismatches.append(
                    "prepared resolved_dataset_config.yaml uses a different split "
                    f"contract: requested={normalized_requested}, prepared={normalized_resolved}"
                )
        requested_protocol = requested.get("protocol_id")
        resolved_protocol = resolved.get("protocol_id")
        if (
            requested_protocol is not None
            and resolved_protocol is not None
            and str(requested_protocol) != str(resolved_protocol)
        ):
            mismatches.append(
                "prepared protocol_id differs from the requested dataset config: "
                f"requested={requested_protocol!r}, prepared={resolved_protocol!r}"
            )

    frozen = requested.get("frozen_split")
    if isinstance(frozen, Mapping):
        frozen_sizes = frozen.get("split_sizes")
        if isinstance(frozen_sizes, Mapping):
            expected = {split: int(frozen_sizes[split]) for split in paths}
            if actual_sizes != expected:
                mismatches.append(
                    "prepared split sizes differ from frozen_split: "
                    f"expected={expected}, found={actual_sizes}"
                )
        # Prefer a semantic split identity when the benchmark comes from a
        # pinned precomputed artifact.  Pickle-byte SHA-256 values are not a
        # stable dataset identity: they can change across Python/NetworkX or
        # pickle protocol versions even when the graph split is identical.
        frozen_source_indices = frozen.get("source_indices")
        frozen_ego_hashes = frozen.get("ego_content_sha256_v1")
        if isinstance(frozen_ego_hashes, Mapping):
            loaded_splits = (
                {split: load_pickle(path) for split, path in paths.items()}
                if splits is None else splits
            )
            observed = _ego_split_content_sha256(loaded_splits)
            for split in paths:
                if frozen_ego_hashes.get(split) != observed[split]:
                    mismatches.append(
                        f"{split} ego graph content differs from frozen_split: "
                        f"expected={frozen_ego_hashes.get(split)}, found={observed[split]}"
                    )
        elif isinstance(frozen_source_indices, Mapping):
            if splits is None:
                loaded_splits = {
                    split: load_pickle(path) for split, path in paths.items()
                }
            else:
                loaded_splits = {
                    split: list(splits.get(split, [])) for split in paths
                }
            observed_source_indices = _source_index_contract(loaded_splits)
            if observed_source_indices is None:
                mismatches.append(
                    "frozen_split.source_indices is configured, but one or more "
                    "prepared graphs do not carry graph['source_index']"
                )
            else:
                for split in paths:
                    expected = [int(v) for v in frozen_source_indices.get(split, [])]
                    found = observed_source_indices[split]
                    if expected != found:
                        mismatches.append(
                            f"{split} source-index sequence differs from frozen_split: "
                            f"expected={expected}, found={found}"
                        )
        else:
            # Legacy fallback.  Raw pickle hashes remain supported for older
            # configs, but new precomputed benchmarks should use source_indices.
            observed_hashes = {split: _sha256(path) for split, path in paths.items()}
            frozen_hashes = frozen.get("split_sha256")
            if isinstance(frozen_hashes, Mapping):
                for split in paths:
                    expected = frozen_hashes.get(split)
                    if expected is not None and str(expected) != observed_hashes[split]:
                        mismatches.append(
                            f"{split} split SHA-256 differs from frozen_split: "
                            f"expected={expected}, found={observed_hashes[split]}"
                        )
            frozen_fingerprint = frozen.get("fingerprint")
            observed_fingerprint = _combined_split_fingerprint(observed_hashes)
            if frozen_fingerprint is not None and str(frozen_fingerprint) != observed_fingerprint:
                mismatches.append(
                    "prepared dataset fingerprint differs from frozen_split: "
                    f"expected={frozen_fingerprint}, found={observed_fingerprint}"
                )

    if mismatches and policy == "error":
        dataset_arg = config_file.stem
        detail = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Prepared dataset does not match the frozen benchmark configuration:\n  - "
            + detail
            + "\nRebuild the prepared split before training/evaluation with:\n  "
            + f"PYTHONPATH=src python scripts/prepare_generic_dataset.py --dataset {dataset_arg} --root {root}"
            + "\nThen retrain/regenerate models whose manifests reference the old split fingerprint."
        )
    if mismatches and policy == "warn":
        print("WARNING: prepared dataset contract mismatch detected:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
    return mismatches

def load_dataset_splits(
    dataset: str,
    *,
    root: str | Path = "outputs/datasets",
    build_if_missing: bool = True,
    config_path: str | Path | None = None,
) -> dict[str, list[nx.Graph]]:
    directory = dataset_dir(dataset, root)
    paths = {split: directory / f"{split}.pkl" for split in ("train", "val", "test")}
    if all(path.exists() for path in paths.values()):
        return {split: load_pickle(path) for split, path in paths.items()}
    if not build_if_missing:
        missing = [str(p) for p in paths.values() if not p.exists()]
        raise FileNotFoundError(f"Missing dataset split files: {missing}")
    if config_path is None:
        config_path = Path("configs/datasets") / f"{dataset}.yaml"
    config = load_yaml(config_path)
    splits = build_splits_from_config(config)
    save_dataset_splits(dataset, splits, config, root)
    return splits
