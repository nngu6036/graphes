#!/usr/bin/env python3
"""Draw molecular or generic graphs from a prepared dataset by name.

The input is a prepared molecular dataset under
``outputs/datasets/<dataset>``. Pass its name with ``--dataset``; the script
then resolves the directory using the same convention as the training and
evaluation commands. ``--count`` graphs are sampled without replacement;
``--seed`` makes the selection reproducible. Use ``--all`` to draw every graph
in the selected split, or combine it with ``--split all`` for the full dataset.
For molecular datasets, both selection and graphlet statistics exclude graphs
that fail validation. Valid molecules must have a nonempty simple topology,
complete atom/bond attributes, and pass RDKit sanitization. Explicit formal
charges are retained and the existing projected QM9 charge inference applies.
The script reports exclusions and rejects counts larger than the valid pool.

Main features
-------------
- resolve a prepared molecular dataset by name
- select its train, validation, or test split explicitly
- optionally combine all three prepared splits
- draw a random sample without replacement
- draw every graph with ``--all``
- arrange molecules in a row x col grid
- if the range is larger than row * col, continue on the next figure/page
- automatically use molecular rendering when atom/bond attributes are present
- otherwise draw a deterministic generic node-link diagram
- write one multipage PDF, including graphlet drawings, with ``--output file.pdf``

Examples
--------
Draw one random molecule::

    PYTHONPATH=src python scripts/draw_dataset.py \
      --dataset qm9_attributed --split test \
      --count 1 --seed 42 \
      --row 1 --col 1 \
      --output outputs/qm9_random.png

Draw 16 random molecules, 4 per row and 3 rows per page::

    PYTHONPATH=src python scripts/draw_dataset.py \
      --dataset qm9_attributed --split test \
      --count 16 --seed 42 \
      --row 3 --col 4 \
      --output outputs/qm9_random_16.png

If more than 12 molecules are requested in the example above, the script saves
multiple files such as:
- outputs/qm9_random_16_page_001.png
- outputs/qm9_random_16_page_002.png

Draw eight generic community graphs::

    PYTHONPATH=src python scripts/draw_dataset.py \
      --dataset community_small --split test \
      --count 8 --seed 42 \
      --row 2 --col 4 \
      --output outputs/community_small_random_8.png

Draw the complete Community-small dataset across all prepared splits::

    PYTHONPATH=src python scripts/draw_dataset.py \
      --dataset community_small --split all --all \
      --row 4 --col 5 \
      --k-min 3 --k-max 5 \
      --output outputs/community_small_all.png

When both ``--k-min`` and ``--k-max`` are supplied, the script also writes
frequency-sorted drawings of induced simple-cycle graphlets, counted across
the full train/validation/test dataset regardless of the drawing selection.
For molecular datasets, counts and normalization use only valid molecules.
Cycles are distinguished by atomic numbers and bond types, including their
arrangement around the ring; equivalent rotations and reflections are grouped.
Drawings show element labels and bond types. Generic graphlets use topology only.
For example, the command above creates
``outputs/community_small_all_graphlet_histogram.png``
and a JSON sidecar containing the raw counts and normalization details.

To collect 1,024 random molecules and the full-dataset cycle graphlet drawings
in one PDF::

    PYTHONPATH=src python scripts/draw_dataset.py \
      --dataset qm9_attributed --split all --count 1024 --seed 42 \
      --row 4 --col 4 --k-min 3 --k-max 5 \
      --output outputs/qm9_random_1024.pdf

PDF output appends the graphlet drawings after the graph pages and writes
``outputs/qm9_random_1024_graphlet_histogram.json`` with the counts. An explicit
``--graphlet-output`` additionally exports the graphlet drawings as PNG or PDF.
All observed types are drawn, with at most six graphlets per page; multiple PNG
pages receive numbered filenames.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import networkx as nx

from grapher.data.statistics import resolve_prepared_dataset
from grapher.rewiring_mlp.molecular.graph_io import nx_to_rdkit_mol
from grapher.utils.io import load_pickle, save_json
from grapher.utils.motifs import canonicalize_attributed_simple_cycle


ATOM_COLOURS_HEX: Mapping[str, str] = {
    "H": "#D5D8DC",
    "C": "#7F8C8D",
    "N": "#3498DB",
    "O": "#E74C3C",
    "F": "#2ECC71",
    "Cl": "#27AE60",
    "Br": "#A04000",
    "I": "#7D3C98",
    "P": "#E67E22",
    "S": "#F1C40F",
    "OTHER": "#AF7AC5",
}

BOND_COLOURS_HEX: Mapping[str, str] = {
    "SINGLE": "#5D6D7E",
    "DOUBLE": "#F39C12",
    "TRIPLE": "#8E44AD",
    "AROMATIC": "#16A085",
    "OTHER": "#34495E",
}

BOND_LABELS: Mapping[str, str] = {
    "SINGLE": "single",
    "DOUBLE": "double",
    "TRIPLE": "triple",
    "AROMATIC": "aromatic",
    "OTHER": "other",
}

BOND_TYPE_KEYS = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE", 4: "AROMATIC"}
GRAPHLETS_PER_PAGE = 6

GENERIC_NODE_COLOURS = (
    "#2563EB",
    "#DC2626",
    "#16A34A",
    "#9333EA",
    "#EA580C",
    "#0891B2",
    "#CA8A04",
    "#4F46E5",
)


@dataclass(frozen=True)
class MoleculeInfo:
    source: str
    name: str
    smiles: str
    dataset_index: Optional[int] = None
    source_index: Optional[int] = None
    index_label: Optional[str] = None


@dataclass(frozen=True)
class LoadedItem:
    info: MoleculeInfo
    mol: Any | None = None
    graph: nx.Graph | None = None
    render_mode: str = "molecule"
    error: str | None = None


@dataclass(frozen=True)
class CycleGraphletFrequency:
    order: int
    count: int
    possible_subsets: int
    frequency: float
    subset_rate: float
    canonical_key: str | None = None
    node_types: tuple[int, ...] = ()
    edge_types: tuple[int, ...] = ()


def _hex_to_rgb255(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex colour, received {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _hex_to_rgb01(value: str) -> Tuple[float, float, float]:
    rgb = _hex_to_rgb255(value)
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def _python_scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _python_scalar(value[0], default)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _safe_smiles(mol: Any) -> str:
    from rdkit import Chem

    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return "<unavailable>"


def _load_prepared_dataset_split(
    dataset: str,
    root: str | Path,
    split: str,
) -> tuple[list[Any], Path, str]:
    """Resolve and load one prepared split without building or downloading data."""

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown prepared dataset split: {split!r}")
    resolved = resolve_prepared_dataset(dataset, root=root)
    split_path = resolved.directory / f"{split}.pkl"
    payload = load_pickle(split_path)
    if not isinstance(payload, (list, tuple)):
        raise TypeError(
            f"Prepared dataset split {split_path} must contain a list of "
            f"NetworkX graphs, not {type(payload).__name__}."
        )
    return list(payload), split_path, resolved.serialized_name


def _load_prepared_dataset_selection(
    dataset: str,
    root: str | Path,
    split: str,
) -> tuple[list[Any], Path, str, list[tuple[str, int]]]:
    """Load one split or the complete disjoint train/val/test dataset."""

    if split != "all":
        graphs, split_path, dataset_name = _load_prepared_dataset_split(
            dataset, root, split
        )
        locations = [(split, index) for index in range(len(graphs))]
        return graphs, split_path, dataset_name, locations

    all_graphs: list[Any] = []
    locations: list[tuple[str, int]] = []
    dataset_name: str | None = None
    dataset_directory: Path | None = None
    for selected_split in ("train", "val", "test"):
        graphs, split_path, resolved_name = _load_prepared_dataset_split(
            dataset, root, selected_split
        )
        if dataset_name is not None and resolved_name != dataset_name:
            raise RuntimeError("Prepared dataset name changed while loading splits.")
        dataset_name = resolved_name
        dataset_directory = split_path.parent
        all_graphs.extend(graphs)
        locations.extend(
            (selected_split, index) for index in range(len(graphs))
        )
    assert dataset_name is not None and dataset_directory is not None
    return all_graphs, dataset_directory, dataset_name, locations


def _sample_graph_indices(dataset_size: int, count: int, seed: int) -> list[int]:
    """Sample distinct split-local graph indices reproducibly."""

    if count <= 0:
        raise ValueError("--count must be positive")
    if count > dataset_size:
        raise ValueError(
            f"--count {count} exceeds the selected split size {dataset_size}"
        )
    return random.Random(seed).sample(range(dataset_size), count)


def _select_graph_indices(
    dataset_size: int,
    *,
    count: int,
    seed: int,
    draw_all: bool,
) -> list[int]:
    if draw_all:
        return list(range(dataset_size))
    return _sample_graph_indices(dataset_size, count, seed)


def _first_attribute(data: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    for name in names:
        if name in data:
            return data[name]
    return None


def _is_molecular_graph(graph: Any) -> bool:
    """Return whether every node and edge carries molecular attributes."""

    if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
        return False
    for _node, data in graph.nodes(data=True):
        value = _first_attribute(
            data,
            ("atomic_num", "atomic_number", "atom_type", "z"),
        )
        if value is None:
            return False
        try:
            atomic_number = int(_python_scalar(value))
        except (TypeError, ValueError):
            return False
        if atomic_number < 1 or atomic_number > 118:
            return False
    for u, v, data in graph.edges(data=True):
        value = _first_attribute(data, ("bond_type", "edge_type", "bond_order"))
        if value is None:
            return False
        try:
            _prepared_bond_type(value, edge=(u, v))
        except ValueError:
            return False
    return True


def _prepared_source_index(graph: nx.Graph) -> int | None:
    source_index = _python_scalar(graph.graph.get("source_index"))
    try:
        return int(source_index) if source_index is not None else None
    except (TypeError, ValueError):
        return None


def _generic_graph_info(
    graph: Any,
    index: int,
    dataset_name: str,
    split: str,
) -> MoleculeInfo:
    if not isinstance(graph, nx.Graph):
        raise TypeError(
            f"{dataset_name}/{split}[{index}] is not a NetworkX graph "
            f"({type(graph).__name__})."
        )
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    if node_count == 0:
        component_count = 0
    elif graph.is_directed():
        component_count = nx.number_weakly_connected_components(graph)
    else:
        component_count = nx.number_connected_components(graph)
    index_label = f"{dataset_name}/{split}[{index}]"
    name = str(graph.graph.get("name", index_label))
    caption = (
        f"nodes={node_count}, edges={edge_count}, components={component_count}"
    )
    return MoleculeInfo(
        source=f"prepared dataset {dataset_name}/{split}",
        name=name,
        smiles=caption,
        dataset_index=index,
        source_index=_prepared_source_index(graph),
        index_label=index_label,
    )


def _prepared_bond_type(value: Any, *, edge: tuple[Any, Any]) -> int:
    raw = _python_scalar(value)
    try:
        numeric = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Edge {edge!r} has invalid molecular bond type {raw!r}."
        ) from exc
    if abs(numeric - 1.5) <= 1.0e-8:
        return 4
    rounded = int(round(numeric))
    if abs(numeric - rounded) > 1.0e-8 or rounded not in {1, 2, 3, 4}:
        raise ValueError(
            f"Edge {edge!r} has unsupported molecular bond type {raw!r}."
        )
    return rounded


def _normalized_molecular_graph(graph: Any, *, label: str) -> nx.Graph:
    """Normalize atom and bond types identically for validation and graphlets."""
    if not isinstance(graph, nx.Graph):
        raise TypeError(
            f"{label} is not a NetworkX graph ({type(graph).__name__})."
        )
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{label} has no atoms.")
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError(f"{label} must be a simple undirected molecular graph.")

    normalized = nx.Graph()
    node_map: dict[Any, int] = {}
    for normalized_index, (node, data) in enumerate(graph.nodes(data=True)):
        atomic_number = _first_attribute(
            data,
            ("atomic_num", "atomic_number", "atom_type", "z"),
        )
        if atomic_number is None:
            raise ValueError(
                f"Node {node!r} in {label} is missing "
                "atomic_num/atom_type. Use an attributed molecular dataset "
                "such as 'qm9_attributed', not a topology-only dataset."
            )
        try:
            numeric = float(_python_scalar(atomic_number))
            atomic_number = int(numeric)
            if numeric != atomic_number or not 1 <= atomic_number <= 118:
                raise ValueError("Atomic numbers must be integers from 1 to 118.")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Node {node!r} in {label} has invalid "
                f"atomic number {atomic_number!r}."
            ) from exc
        node_map[node] = normalized_index
        normalized.add_node(normalized_index, atomic_num=atomic_number)
        if data.get("formal_charge") is not None:
            raw_charge = float(_python_scalar(data["formal_charge"]))
            charge = int(raw_charge)
            if raw_charge != charge:
                raise ValueError(f"Node {node!r} in {label} has a non-integer formal charge.")
            normalized.nodes[normalized_index]["formal_charge"] = charge

    for u, v, data in graph.edges(data=True):
        raw_bond_type = _first_attribute(
            data,
            ("bond_type", "edge_type", "bond_order"),
        )
        if raw_bond_type is None:
            raise ValueError(
                f"Edge {(u, v)!r} in {label} is missing "
                "bond_type. Use an attributed molecular dataset such as "
                "'qm9_attributed', not a topology-only dataset."
            )
        normalized.add_edge(
            node_map[u],
            node_map[v],
            bond_type=_prepared_bond_type(raw_bond_type, edge=(u, v)),
        )

    return normalized


def _validated_molecule(graph: Any, *, label: str) -> Any:
    """Require successful RDKit sanitization of the normalized molecular graph."""
    return nx_to_rdkit_mol(
        _normalized_molecular_graph(graph, label=label),
        sanitize=True,
        infer_projected_formal_charges=True,
    )


def _load_from_prepared_graph(
    graph: Any,
    index: int,
    dataset_name: str,
    split: str,
) -> Tuple[Any, MoleculeInfo]:
    """Convert one valid prepared molecular graph and retain its original indices."""
    index_label = f"{dataset_name}/{split}[{index}]"
    mol = _validated_molecule(graph, label=index_label)
    name = str(graph.graph.get("name", index_label))
    return mol, MoleculeInfo(
        source=f"prepared dataset {dataset_name}/{split}",
        name=name,
        smiles=_safe_smiles(mol),
        dataset_index=index,
        source_index=_prepared_source_index(graph),
        index_label=index_label,
    )


def _is_molecular_dataset(graphs: Sequence[Any], dataset_name: str) -> bool:
    """Detect molecular intent even when some records have incomplete attributes."""
    if dataset_name.lower() in {"qm9", "qm9_attributed", "zinc", "zinc_attributed", "zinc250k"}:
        return True
    for graph in graphs:
        if not isinstance(graph, nx.Graph):
            continue
        if any(
            any(key in data for key in ("atomic_num", "atomic_number", "atom_type", "z"))
            for _, data in graph.nodes(data=True)
        ) or any(
            "bond_type" in data or "bond_order" in data
            for _, _, data in graph.edges(data=True)
        ):
            return True
    return False


def _valid_molecular_indices(graphs: Sequence[Any], *, dataset_label: str) -> list[int]:
    # Import before filtering so missing RDKit is an environment error, not an
    # apparent dataset containing no valid molecules. Suppress per-record RDKit
    # diagnostics while reporting the aggregate result below.
    from rdkit import rdBase

    print(f"Validating {len(graphs):,} molecular graphs in {dataset_label}...", flush=True)
    valid: list[int] = []
    with rdBase.BlockLogs():
        for index, graph in enumerate(graphs):
            try:
                _validated_molecule(graph, label=f"{dataset_label}[{index}]")
            except (ValueError, TypeError, RuntimeError, OverflowError):
                pass
            else:
                valid.append(index)
            if (index + 1) % 10000 == 0:
                print(f"Validated {index + 1:,}/{len(graphs):,}: valid={len(valid):,}", flush=True)
    print(
        f"Molecular validity: valid={len(valid):,} "
        f"excluded={len(graphs) - len(valid):,} total={len(graphs):,}",
        flush=True,
    )
    return valid


def _prepare_molecule(mol: Any, show_hydrogens: bool, atom_indices: bool, bond_labels: bool) -> Any:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.Mol(mol)
    if show_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            mol = Chem.RemoveHs(mol, sanitize=False)
            mol.UpdatePropertyCache(strict=False)

    if mol.GetNumAtoms() == 0:
        raise ValueError("The molecule has no drawable atoms after hydrogen handling")

    rdDepictor.Compute2DCoords(mol, canonOrient=True)

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom.SetProp("atomLabel", f"{symbol}{atom.GetIdx()}" if atom_indices else symbol)

    if bond_labels:
        for bond in mol.GetBonds():
            key = str(bond.GetBondType()).upper()
            bond.SetProp("bondNote", BOND_LABELS.get(key, "other"))

    return mol


def _highlight_maps(mol: Any):
    atoms = []
    atom_colours = {}
    atom_radii = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        atoms.append(idx)
        atom_colours[idx] = _hex_to_rgb01(ATOM_COLOURS_HEX.get(symbol, ATOM_COLOURS_HEX["OTHER"]))
        atom_radii[idx] = 0.34 if symbol != "H" else 0.25

    bonds = []
    bond_colours = {}
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        key = str(bond.GetBondType()).upper()
        bonds.append(idx)
        bond_colours[idx] = _hex_to_rgb01(BOND_COLOURS_HEX.get(key, BOND_COLOURS_HEX["OTHER"]))

    return atoms, atom_colours, atom_radii, bonds, bond_colours


def _configure_drawer(drawer: Any) -> None:
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fillHighlights = True
    opts.atomHighlightsAreCircles = True
    opts.continuousHighlight = False
    opts.highlightBondWidthMultiplier = 16
    opts.bondLineWidth = 2.2
    opts.annotationFontScale = 0.58
    opts.padding = 0.06
    opts.addAtomIndices = False


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    words = text.split()
    if len(words) <= 1:
        return [text[: width - 3] + "..."]

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
        if len(lines) == 2:
            break
    if current and len(lines) < 2:
        lines.append(current)
    if len(lines) > 2:
        lines = lines[:2]
    if len(lines) == 2 and len(" ".join(words)) > sum(len(x) for x in lines) + 1:
        if len(lines[1]) > width - 3:
            lines[1] = lines[1][: width - 3] + "..."
        else:
            lines[1] = lines[1] + "..."
    return lines


def _draw_source_index_badge(
    draw: Any,
    info: MoleculeInfo,
    panel_width: int,
    panel_height: int,
    font: Any,
) -> None:
    if info.source_index is None:
        return
    source_text = f"source {info.source_index}"
    bbox = draw.textbbox((0, 0), source_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x1 = panel_width - text_width - 18
    y1 = panel_height - text_height - 12
    draw.rounded_rectangle(
        (x1 - 8, y1 - 4, x1 + text_width + 8, y1 + text_height + 4),
        radius=8,
        fill="#F3F4F6",
        outline="#D1D5DB",
    )
    draw.text((x1, y1), source_text, fill="#374151", font=font)


def _generic_node_category(data: Mapping[str, Any]) -> str | None:
    value = _first_attribute(
        data,
        ("community", "block", "node_label", "label", "type", "group"),
    )
    if value is None:
        return None
    scalar = _python_scalar(value)
    return str(scalar)


def _render_generic_graph_panel(
    graph: nx.Graph,
    info: MoleculeInfo,
    panel_width: int,
    panel_height: int,
    show_title: bool,
    layout_seed: int,
) -> Any:
    """Render a generic NetworkX graph without importing RDKit."""

    from PIL import Image, ImageDraw

    bg = Image.new("RGB", (panel_width, panel_height), "white")
    draw = ImageDraw.Draw(bg)
    draw.rounded_rectangle(
        (2, 2, panel_width - 3, panel_height - 3),
        radius=12,
        outline="#D0D7DE",
        width=2,
    )
    title_font = _load_font(18, bold=True)
    body_font = _load_font(14)
    small_font = _load_font(11)
    index_label = info.index_label or f"graph {info.dataset_index}"
    top_pad = 14
    title_height = 26 if show_title else 0
    caption_height = 52

    if show_title:
        draw.text((14, top_pad), index_label, fill="#111827", font=title_font)

    nodes = list(graph.nodes())
    if not nodes:
        message = "Empty graph"
        bbox = draw.textbbox((0, 0), message, font=body_font)
        draw.text(
            (
                (panel_width - (bbox[2] - bbox[0])) / 2,
                (panel_height - caption_height) / 2,
            ),
            message,
            fill="#6B7280",
            font=body_font,
        )
    else:
        if len(nodes) == 1:
            positions = {nodes[0]: (0.0, 0.0)}
        elif len(nodes) <= 200:
            positions = nx.spring_layout(graph, seed=int(layout_seed), iterations=50)
        else:
            positions = nx.circular_layout(graph)

        left = 26.0
        right = float(panel_width - 26)
        top = float(top_pad + title_height + 10)
        bottom = float(panel_height - caption_height - 10)
        x_values = [float(positions[node][0]) for node in nodes]
        y_values = [float(positions[node][1]) for node in nodes]
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        def scale(
            value: float,
            low: float,
            high: float,
            start: float,
            end: float,
        ) -> float:
            if abs(high - low) <= 1.0e-12:
                return (start + end) / 2.0
            return start + (value - low) * (end - start) / (high - low)

        points = {
            node: (
                scale(float(positions[node][0]), min_x, max_x, left, right),
                scale(float(positions[node][1]), min_y, max_y, bottom, top),
            )
            for node in nodes
        }
        for u, v in graph.edges():
            x1, y1 = points[u]
            x2, y2 = points[v]
            if u == v:
                draw.ellipse(
                    (x1 - 9, y1 - 15, x1 + 9, y1 + 3),
                    outline="#64748B",
                    width=2,
                )
            else:
                draw.line((x1, y1, x2, y2), fill="#94A3B8", width=2)

        categories = {
            node: _generic_node_category(graph.nodes[node]) for node in nodes
        }
        category_values = sorted(
            {value for value in categories.values() if value is not None}
        )
        category_colours = {
            value: GENERIC_NODE_COLOURS[index % len(GENERIC_NODE_COLOURS)]
            for index, value in enumerate(category_values)
        }
        radius = max(4, min(10, int(32 / math.sqrt(max(len(nodes), 1)))))
        show_node_labels = len(nodes) <= 30
        for node in nodes:
            x, y = points[node]
            colour = category_colours.get(categories[node], GENERIC_NODE_COLOURS[0])
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=colour,
                outline="#1E293B",
                width=1,
            )
            if show_node_labels:
                label = str(node)
                bbox = draw.textbbox((0, 0), label, font=small_font)
                draw.text(
                    (
                        x - (bbox[2] - bbox[0]) / 2,
                        y - (bbox[3] - bbox[1]) / 2,
                    ),
                    label,
                    fill="white",
                    font=small_font,
                )

    caption_y = panel_height - caption_height
    draw.text((14, caption_y), info.name, fill="#111827", font=body_font)
    for line_index, line in enumerate(_wrap_text(info.smiles, 44)):
        draw.text(
            (14, caption_y + 18 + 15 * line_index),
            line,
            fill="#4B5563",
            font=small_font,
        )
    _draw_source_index_badge(draw, info, panel_width, panel_height, small_font)
    return bg


def _render_molecule_panel(
    loaded: LoadedItem,
    panel_width: int,
    panel_height: int,
    show_title: bool,
) -> Any:
    from PIL import Image, ImageDraw

    bg = Image.new("RGB", (panel_width, panel_height), "white")
    draw = ImageDraw.Draw(bg)
    draw.rounded_rectangle((2, 2, panel_width - 3, panel_height - 3), radius=12, outline="#D0D7DE", width=2)

    title_font = _load_font(18, bold=True)
    body_font = _load_font(14)
    small_font = _load_font(12)

    top_pad = 14
    caption_h = 56
    title_h = 26 if show_title else 0
    molecule_h = panel_height - title_h - caption_h - 20
    molecule_h = max(molecule_h, 120)
    index_label = (
        loaded.info.index_label or f"QM9 index {loaded.info.dataset_index}"
    )

    if loaded.error or loaded.mol is None:
        y = 20
        draw.text((16, y), index_label, fill="#B42318", font=title_font)
        y += 34
        for line in _wrap_text(loaded.error or "Unknown loading error", 42):
            draw.text((16, y), line, fill="#5F2120", font=body_font)
            y += 20
        return bg

    from rdkit.Chem.Draw import rdMolDraw2D

    if show_title:
        draw.text((14, top_pad), index_label, fill="#111827", font=title_font)

    atoms, atom_colours, atom_radii, bonds, bond_colours = _highlight_maps(loaded.mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(panel_width - 20, molecule_h)
    _configure_drawer(drawer)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        loaded.mol,
        legend="",
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors=atom_colours,
        highlightBondColors=bond_colours,
        highlightAtomRadii=atom_radii,
        kekulize=False,
    )
    drawer.FinishDrawing()
    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    bg.paste(mol_img, (10, top_pad + title_h))

    caption_y = panel_height - caption_h
    caption = loaded.info.smiles
    draw.text((14, caption_y), loaded.info.name, fill="#111827", font=body_font)
    for i, line in enumerate(_wrap_text(caption, 38)):
        draw.text((14, caption_y + 18 + 16 * i), line, fill="#4B5563", font=small_font)

    _draw_source_index_badge(
        draw,
        loaded.info,
        panel_width,
        panel_height,
        small_font,
    )

    return bg


def _draw_page_legend(canvas: Any, y0: int, page_width: int, show_hydrogens: bool) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(15, bold=True)
    body_font = _load_font(13)

    draw.line((20, y0, page_width - 20, y0), fill="#E5E7EB", width=2)
    y = y0 + 12
    draw.text((20, y), "Node / atom types", fill="#111827", font=title_font)
    x = 170
    atom_keys = ["H", "C", "N", "O", "F"] if show_hydrogens else ["C", "N", "O", "F"]
    for symbol in atom_keys:
        draw.ellipse((x, y - 1, x + 16, y + 15), fill=ATOM_COLOURS_HEX[symbol], outline="#4B5563", width=1)
        draw.text((x + 22, y - 2), symbol, fill="#111827", font=body_font)
        x += 58

    x += 20
    draw.text((x, y), "Edge / bond types", fill="#111827", font=title_font)
    x += 135
    for key in ("SINGLE", "DOUBLE", "TRIPLE"):
        colour = BOND_COLOURS_HEX[key]
        line_count = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}[key]
        offsets = {1: [0], 2: [-3, 3], 3: [-5, 0, 5]}[line_count]
        for offset in offsets:
            draw.line((x, y + 7 + offset, x + 26, y + 7 + offset), fill=colour, width=3)
        draw.text((x + 34, y - 2), BOND_LABELS[key], fill="#111827", font=body_font)
        x += 86


def _draw_generic_page_legend(
    canvas: Any,
    y0: int,
    page_width: int,
    *,
    mixed: bool = False,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(15, bold=True)
    body_font = _load_font(13)
    draw.line((20, y0, page_width - 20, y0), fill="#E5E7EB", width=2)
    y = y0 + 12
    title = "Mixed molecular and generic graphs" if mixed else "Generic graphs"
    draw.text((20, y), title, fill="#111827", font=title_font)
    x = 245 if mixed else 145
    draw.line((x, y + 7, x + 34, y + 7), fill="#94A3B8", width=2)
    draw.ellipse(
        (x + 10, y - 1, x + 26, y + 15),
        fill=GENERIC_NODE_COLOURS[0],
        outline="#1E293B",
    )
    draw.text(
        (x + 46, y - 2),
        "deterministic node-link layout",
        fill="#111827",
        font=body_font,
    )


def _is_induced_cycle(graph: nx.Graph, nodes: Sequence[Any]) -> bool:
    """Return whether the selected nodes induce exactly one simple cycle."""

    if len(nodes) < 3:
        return False
    simple_graph = nx.Graph(graph)
    adjacency = {node: set(simple_graph.neighbors(node)) for node in nodes}
    return _is_induced_cycle_from_adjacency(adjacency, nodes)


def _is_induced_cycle_from_adjacency(
    adjacency: Mapping[Any, set[Any]],
    nodes: Sequence[Any],
) -> bool:
    """Fast exact-cycle check used inside the subset enumeration loop."""

    if len(nodes) < 3:
        return False
    selected = set(nodes)
    for node in nodes:
        neighbours = adjacency[node]
        if node in neighbours or len(neighbours & selected) != 2:
            return False

    start = nodes[0]
    visited = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for neighbour in adjacency[current] & selected:
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return len(visited) == len(nodes)


def _cycle_graphlet_histogram(
    graphs: Sequence[nx.Graph],
    *,
    k_min: int,
    k_max: int,
    molecular: bool = False,
) -> list[CycleGraphletFrequency]:
    """Count induced cycles, respecting atom/bond types for molecular datasets."""

    if k_min < 3:
        raise ValueError("Cycle graphlets require --graphlet-k-min >= 3.")
    if k_max < k_min:
        raise ValueError("--graphlet-k-max must be >= --graphlet-k-min.")
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError("Graphlet histograms require NetworkX graphs.")

    counts: dict[tuple[int, str | None], int] = (
        {} if molecular else {(order, None): 0 for order in range(k_min, k_max + 1)}
    )
    possible = {order: 0 for order in range(k_min, k_max + 1)}
    for graph in graphs:
        simple_graph = (
            _normalized_molecular_graph(graph, label="graphlet source")
            if molecular else nx.Graph(graph)
        )
        nodes = tuple(simple_graph.nodes())
        adjacency = {
            node: set(simple_graph.neighbors(node)) for node in nodes
        }
        for order in possible:
            if len(nodes) < order:
                continue
            possible[order] += math.comb(len(nodes), order)
            for subset in itertools.combinations(nodes, order):
                if not _is_induced_cycle_from_adjacency(adjacency, subset):
                    continue
                key = (
                    canonicalize_attributed_simple_cycle(
                        simple_graph.subgraph(subset),
                        node_label_attr="atomic_num", edge_label_attr="bond_type",
                    )
                    if molecular else None
                )
                counts[order, key] = counts.get((order, key), 0) + 1

    total_cycles = sum(counts.values())
    rows = []
    for (order, key), count in counts.items():
        # ATTR_CYCLE_V1 stores [node type, outgoing bond type] around the
        # canonical cycle. Normalization above guarantees integer type tokens.
        encoded = json.loads(key.split("|", 1)[1]) if key is not None else []
        node_types = tuple(int(node.split(":", 1)[1]) for node, _ in encoded)
        edge_types = tuple(int(edge.split(":", 1)[1]) for _, edge in encoded)
        rows.append(CycleGraphletFrequency(
            order=order,
            count=count,
            possible_subsets=possible[order],
            frequency=(count / total_cycles if total_cycles else 0.0),
            subset_rate=(
                count / possible[order] if possible[order] else 0.0
            ),
            canonical_key=key, node_types=node_types, edge_types=edge_types,
        ))
    return sorted(rows, key=lambda row: (-row.frequency, row.order, row.canonical_key or ""))


def _draw_typed_cycle_bond(draw: Any, start: tuple[float, float], end: tuple[float, float], bond_type: int) -> None:
    colour = BOND_COLOURS_HEX[BOND_TYPE_KEYS[bond_type]]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if bond_type == 4:
        for distance in range(0, math.ceil(length), 10):
            left, right = distance / length, min(distance + 6, length) / length
            draw.line(
                (start[0] + left * dx, start[1] + left * dy,
                 start[0] + right * dx, start[1] + right * dy),
                fill=colour, width=3,
            )
        return
    offsets = {1: (0,), 2: (-3, 3), 3: (-5, 0, 5)}[bond_type]
    for offset in offsets:
        shift_x, shift_y = -dy * offset / length, dx * offset / length
        draw.line(
            (start[0] + shift_x, start[1] + shift_y,
             end[0] + shift_x, end[1] + shift_y),
            fill=colour, width=3,
        )


def _render_cycle_graphlet_histogram(
    rows: Sequence[CycleGraphletFrequency],
    *,
    dataset_label: str,
    graph_count: int,
    molecular: bool = False,
    page_index: int = 0,
    total_pages: int = 1,
) -> Any:
    """Draw each cycle, including molecular types, with dataset-wide frequencies."""

    from PIL import Image, ImageDraw

    width = 1100
    row_height = 130
    top = 112
    molecular = molecular or any(row.canonical_key is not None for row in rows)
    bottom = 100 if molecular else 68
    height = max(320, top + bottom + row_height * max(len(rows), 1))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(24, bold=True)
    body_font = _load_font(15)
    label_font = _load_font(16, bold=True)
    small_font = _load_font(13)
    atom_font = _load_font(13, bold=True)
    title = f"{dataset_label}: induced {'typed ' if molecular else ''}cycle graphlets"
    if total_pages > 1:
        title += f" | page {page_index + 1}/{total_pages}"

    draw.text(
        (28, 22),
        title,
        fill="#111827",
        font=title_font,
    )
    draw.text(
        (28, 58),
        (
            f"graphs={graph_count} | only chordless induced subgraphs Ck | "
            "sorted by cycle-share frequency"
        ),
        fill="#4B5563",
        font=body_font,
    )

    label_x = 34
    value_x = 260
    colours = ("#2563EB", "#0891B2", "#16A34A", "#9333EA", "#EA580C")

    for index, row in enumerate(rows):
        y = top + index * row_height
        draw.text((label_x, y + 8), f"C{row.order}", fill="#111827", font=label_font)
        points = [
            (
                170 + 46 * math.cos(2 * math.pi * node / row.order - math.pi / 2),
                y + 58 + 46 * math.sin(2 * math.pi * node / row.order - math.pi / 2),
            )
            for node in range(row.order)
        ]
        if row.canonical_key is not None:
            from rdkit import Chem

            for edge_index, bond_type in enumerate(row.edge_types):
                _draw_typed_cycle_bond(draw, points[edge_index], points[(edge_index + 1) % row.order], bond_type)
            symbols = [Chem.GetPeriodicTable().GetElementSymbol(value) for value in row.node_types]
            for (x_node, y_node), symbol in zip(points, symbols):
                draw.ellipse(
                    (x_node - 12, y_node - 12, x_node + 12, y_node + 12),
                    fill=ATOM_COLOURS_HEX.get(symbol, ATOM_COLOURS_HEX["OTHER"]),
                    outline="#1E293B",
                )
                draw.text((x_node, y_node), symbol, anchor="mm", fill="#111827", font=atom_font)
            draw.text(
                (label_x, y + 32), f"#{page_index * GRAPHLETS_PER_PAGE + index + 1}",
                fill="#6B7280", font=small_font,
            )
            draw.text((value_x, y + 51), "Atoms: " + " / ".join(symbols), fill="#4B5563", font=small_font)
            draw.text(
                (value_x, y + 74),
                "Bonds: " + " / ".join(BOND_LABELS[BOND_TYPE_KEYS[value]] for value in row.edge_types),
                fill="#4B5563", font=small_font,
            )
        else:
            draw.line(points + [points[0]], fill="#64748B", width=3)
            for x_node, y_node in points:
                draw.ellipse(
                    (x_node - 6, y_node - 6, x_node + 6, y_node + 6),
                    fill=colours[index % len(colours)], outline="#1E293B",
                )
        draw.text(
            (value_x, y + 1),
            f"frequency={row.frequency:.6f}  count={row.count:,}",
            fill="#111827",
            font=body_font,
        )
        draw.text(
            (value_x, y + 25),
            (
                f"induced-subset rate={row.subset_rate:.6g}  "
                f"eligible subsets={row.possible_subsets:,}"
            ),
            fill="#6B7280",
            font=small_font,
        )

    if not rows:
        draw.text((28, top), "No induced cycle graphlets found." if molecular else "No graphlet orders requested.", fill="#6B7280", font=body_font)
    draw.line((28, height - bottom + 24, width - 28, height - bottom + 24), fill="#D1D5DB", width=2)
    if molecular:
        for index, (bond_type, key) in enumerate(BOND_TYPE_KEYS.items()):
            x = 28 + index * 250
            _draw_typed_cycle_bond(draw, (x, height - 56), (x + 38, height - 56), bond_type)
            draw.text(
                (x + 48, height - 64), BOND_LABELS[key] + (" (dashed)" if bond_type == 4 else ""),
                fill="#4B5563", font=small_font,
            )
    draw.text(
        (28, height - 32),
        "frequency = graphlet count / total induced-cycle count over all requested k",
        fill="#6B7280",
        font=small_font,
    )
    return canvas


def _graphlet_histogram_output_path(
    graph_output: Path,
    configured: Path | None,
) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    if graph_output.suffix.lower() == ".pdf":
        return graph_output
    return graph_output.with_name(f"{graph_output.stem}_graphlet_histogram.png")


def _save_drawing(canvas: Any, output: Path, *, append: bool = False) -> None:
    """Write one drawing, appending PDF pages without retaining all canvases."""

    if output.suffix.lower() == ".pdf":
        canvas.save(output, format="PDF", append=append, resolution=144.0, quality=95)
    else:
        canvas.save(output)


def _compose_page(
    items: list[LoadedItem],
    row: int,
    col: int,
    panel_width: int,
    panel_height: int,
    page_index: int,
    total_pages: int,
    count: int,
    seed: int,
    dataset_label: str,
    dataset_path: Path,
    show_hydrogens: bool,
    page_title: str | None = None,
) -> Any:
    from PIL import Image, ImageDraw

    outer_pad = 18
    gap = 14
    header_h = 54
    legend_h = 56
    page_width = outer_pad * 2 + col * panel_width + (col - 1) * gap
    page_height = outer_pad * 2 + header_h + row * panel_height + (row - 1) * gap + legend_h

    canvas = Image.new("RGB", (page_width, page_height), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(22, bold=True)
    body_font = _load_font(13)

    title = page_title or f"{dataset_label} random sample: n={count}, seed={seed}"
    if total_pages > 1:
        title += f"  |  page {page_index + 1}/{total_pages}"
    draw.text((outer_pad, outer_pad), title, fill="#111827", font=title_font)
    draw.text(
        (outer_pad, outer_pad + 28),
        f"Dataset: {dataset_path}",
        fill="#6B7280",
        font=body_font,
    )

    start_y = outer_pad + header_h
    for i, item in enumerate(items):
        r = i // col
        c = i % col
        x = outer_pad + c * (panel_width + gap)
        y = start_y + r * (panel_height + gap)
        if item.error is not None or item.render_mode == "molecule":
            panel = _render_molecule_panel(
                item,
                panel_width,
                panel_height,
                show_title=True,
            )
        elif item.graph is not None:
            panel = _render_generic_graph_panel(
                item.graph,
                item.info,
                panel_width,
                panel_height,
                show_title=True,
                layout_seed=(seed + int(item.info.dataset_index or 0)) % (2**32),
            )
        else:
            raise ValueError("Generic render item is missing its NetworkX graph.")
        canvas.paste(panel, (x, y))

    render_modes = {item.render_mode for item in items if item.error is None}
    if render_modes == {"molecule"}:
        _draw_page_legend(canvas, page_height - legend_h, page_width, show_hydrogens)
    else:
        _draw_generic_page_legend(
            canvas,
            page_height - legend_h,
            page_width,
            mixed=len(render_modes) > 1,
        )
    return canvas


def _page_output_path(output: Path, page_index: int, total_pages: int) -> Path:
    if total_pages <= 1:
        return output
    return output.with_name(f"{output.stem}_page_{page_index + 1:03d}{output.suffix}")


def _default_output(count: int, seed: int, prefix: str) -> Path:
    return Path(f"outputs/{prefix}_sample_n{count}_seed{seed}.png")


def _default_all_output(prefix: str) -> Path:
    return Path(f"outputs/{prefix}_all.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a sample or every molecular/generic graph from a named "
            "prepared dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Prepared dataset name or configs/datasets/<name>.yaml stem, for "
            "example qm9_attributed."
        ),
    )
    parser.add_argument(
        "--root",
        "--dataset-root",
        dest="root",
        default="outputs/datasets",
        help="Root containing <dataset>/{train,val,test}.pkl.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="test",
        help="Prepared dataset split to draw, or all for train+val+test.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of distinct graphs to sample; molecular datasets use valid molecules only.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Draw every eligible graph in the selected splits (valid molecules for molecular data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to select graphs.",
    )
    parser.add_argument(
        "--row", type=int, default=2, help="Number of graph rows per figure."
    )
    parser.add_argument(
        "--col", type=int, default=4, help="Number of graph columns per figure."
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=360,
        help="Width of each graph panel in pixels.",
    )
    parser.add_argument(
        "--panel-height",
        type=int,
        default=300,
        help="Height of each graph panel in pixels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output .pdf or .png path. PDF collects all graph pages and any "
            "cycle graphlet drawings in one document; PNG creates numbered "
            "files when multiple graph pages are needed."
        ),
    )
    parser.add_argument(
        "--k-min",
        "--graphlet-k-min",
        dest="k_min",
        type=int,
        help=(
            "Minimum induced cycle-graphlet order. Must be provided together "
            "with --k-max and be at least 3. Molecular cycles distinguish "
            "atomic numbers and bond types."
        ),
    )
    parser.add_argument(
        "--k-max",
        "--graphlet-k-max",
        dest="k_max",
        type=int,
        help=(
            "Maximum induced cycle-graphlet order. Must be provided together "
            "with --k-min."
        ),
    )
    parser.add_argument(
        "--graphlet-output",
        type=Path,
        help=(
            "Optional PNG or PDF path for cycle drawings with full-dataset "
            "frequencies. By default they are included in the main PDF, or "
            "saved as <output-stem>_graphlet_histogram.png for PNG output."
        ),
    )
    parser.add_argument(
        "--show-hydrogens",
        action="store_true",
        help="Draw explicit hydrogens in molecular panels; ignored for generic graphs.",
    )
    parser.add_argument(
        "--atom-indices",
        action="store_true",
        help="Append RDKit atom indices in molecular panels.",
    )
    parser.add_argument(
        "--bond-labels",
        action="store_true",
        help="Annotate molecular bonds with single/double/triple/aromatic.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.row <= 0 or args.col <= 0:
        raise ValueError("--row and --col must be positive")
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.panel_width < 180 or args.panel_height < 180:
        raise ValueError("--panel-width and --panel-height should be at least 180")
    graphlet_requested = args.k_min is not None or args.k_max is not None
    if graphlet_requested and (args.k_min is None or args.k_max is None):
        raise ValueError("--k-min and --k-max must be provided together")
    if args.graphlet_output is not None and not graphlet_requested:
        raise ValueError("--graphlet-output requires --k-min and --k-max")
    if graphlet_requested:
        if args.k_min < 3:
            raise ValueError("Cycle graphlets require --k-min >= 3")
        if args.k_max < args.k_min:
            raise ValueError("--k-max must be >= --k-min")
    for option, path in (("--output", args.output), ("--graphlet-output", args.graphlet_output)):
        if path is not None and path.suffix.lower() not in {".png", ".pdf"}:
            raise ValueError(f"{option} must use a .png or .pdf extension")

    print(
        f"Resolving prepared dataset {args.dataset!r} under {args.root}...",
        flush=True,
    )
    # Graphlet statistics use every split. Load that pool once, then keep the
    # drawing selection separate so both consumers share the validity filter.
    loaded_split = "all" if graphlet_requested else args.split
    prepared_graphs, dataset_path, dataset_name, graph_locations = (
        _load_prepared_dataset_selection(
            args.dataset,
            args.root,
            loaded_split,
        )
    )
    molecular_dataset = _is_molecular_dataset(prepared_graphs, dataset_name)
    eligible_indices = (
        _valid_molecular_indices(prepared_graphs, dataset_label=f"{dataset_name}/{loaded_split}")
        if molecular_dataset else list(range(len(prepared_graphs)))
    )
    candidate_indices = [
        index for index in eligible_indices
        if args.split == "all" or graph_locations[index][0] == args.split
    ]
    if molecular_dataset:
        if not candidate_indices:
            raise ValueError(f"No valid molecular graphs remain in {dataset_name}/{args.split}.")
        if not args.all and args.count > len(candidate_indices):
            raise ValueError(
                f"--count {args.count} exceeds the {len(candidate_indices)} valid molecular "
                f"graphs in {dataset_name}/{args.split}; reduce --count or use --all."
            )
    indices = [
        candidate_indices[index] for index in _select_graph_indices(
            len(candidate_indices), count=args.count, seed=args.seed, draw_all=args.all,
        )
    ]
    dataset_label = f"{dataset_name}/{args.split}"
    if molecular_dataset:
        dataset_label += " (valid molecules)"
    if loaded_split == "all" and args.split != "all":
        dataset_path = dataset_path / f"{args.split}.pkl"
    output_prefix = f"{dataset_name}_{args.split}"
    print(
        f"Using prepared dataset: {dataset_name} split={args.split} "
        f"eligible_graphs={len(candidate_indices)} path={dataset_path}",
        flush=True,
    )
    if args.all:
        print(f"Selected every graph: {len(indices)}", flush=True)
    else:
        selected_labels = [
            f"{graph_locations[index][0]}[{graph_locations[index][1]}]"
            for index in indices
        ]
        print(f"Selected graph indices: {selected_labels}", flush=True)

    output = (
        args.output
        or (
            _default_all_output(output_prefix)
            if args.all
            else _default_output(args.count, args.seed, output_prefix)
        )
    ).expanduser().resolve()
    pdf_output = output.suffix.lower() == ".pdf"
    histogram_output = _graphlet_histogram_output_path(output, args.graphlet_output)
    if graphlet_requested and not pdf_output and histogram_output == output:
        raise ValueError("--graphlet-output must differ from the graph PNG output")
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded_items: list[LoadedItem] = []
    for index in indices:
        selected_split, split_index = graph_locations[index]
        try:
            graph = prepared_graphs[index]
            if molecular_dataset:
                mol, info = _load_from_prepared_graph(
                    graph,
                    split_index,
                    dataset_name,
                    selected_split,
                )
                mol = _prepare_molecule(
                    mol,
                    show_hydrogens=args.show_hydrogens,
                    atom_indices=args.atom_indices,
                    bond_labels=args.bond_labels,
                )
                loaded_items.append(
                    LoadedItem(info=info, mol=mol, render_mode="molecule")
                )
            else:
                info = _generic_graph_info(
                    graph,
                    split_index,
                    dataset_name,
                    selected_split,
                )
                loaded_items.append(
                    LoadedItem(info=info, graph=graph, render_mode="generic")
                )
        except Exception as exc:
            index_label = f"{dataset_name}/{selected_split}[{split_index}]"
            print(f"Warning: failed to load {index_label}: {exc}", file=sys.stderr)
            loaded_items.append(
                LoadedItem(
                    info=MoleculeInfo(
                        source=str(dataset_path),
                        name=index_label,
                        smiles="<unavailable>",
                        dataset_index=index,
                        source_index=None,
                        index_label=index_label,
                    ),
                    mol=None,
                    error=str(exc),
                )
            )

    per_page = args.row * args.col
    total_pages = math.ceil(len(loaded_items) / per_page)
    for page_index in range(total_pages):
        start = page_index * per_page
        end = min(start + per_page, len(loaded_items))
        page_items = loaded_items[start:end]
        canvas = _compose_page(
            page_items,
            row=args.row,
            col=args.col,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
            page_index=page_index,
            total_pages=total_pages,
            count=len(indices),
            seed=args.seed,
            dataset_label=dataset_label,
            dataset_path=dataset_path,
            show_hydrogens=args.show_hydrogens,
            page_title=(
                f"{dataset_label}: all graphs (n={len(indices)})"
                if args.all
                else None
            ),
        )
        page_output = output if pdf_output else _page_output_path(output, page_index, total_pages)
        _save_drawing(canvas, page_output, append=pdf_output and page_index > 0)
        if pdf_output:
            canvas.close()
            print(f"Saved PDF graph page {page_index + 1}/{total_pages}: {page_output}", flush=True)
        else:
            print(f"Saved: {page_output}")

    if graphlet_requested:
        histogram_graphs = [prepared_graphs[index] for index in eligible_indices]
        eligible_subsets = sum(
            math.comb(graph.number_of_nodes(), order)
            for graph in histogram_graphs
            for order in range(args.k_min, args.k_max + 1)
            if isinstance(graph, nx.Graph)
            and graph.number_of_nodes() >= order
        )
        print(
            "Counting exact induced cycle graphlets "
            f"C{args.k_min}..C{args.k_max} over "
            f"all {len(histogram_graphs)} graphs (train+val+test), "
            f"{eligible_subsets:,} eligible node subsets...",
            flush=True,
        )
        histogram_rows = _cycle_graphlet_histogram(
            histogram_graphs,
            k_min=args.k_min,
            k_max=args.k_max,
            molecular=molecular_dataset,
        )
        histogram_output.parent.mkdir(parents=True, exist_ok=True)
        graphlet_pages = max(1, math.ceil(len(histogram_rows) / GRAPHLETS_PER_PAGE))
        for page_index in range(graphlet_pages):
            start = page_index * GRAPHLETS_PER_PAGE
            histogram = _render_cycle_graphlet_histogram(
                histogram_rows[start:start + GRAPHLETS_PER_PAGE],
                dataset_label=f"{dataset_name}/all" + (" (valid molecules)" if molecular_dataset else ""),
                graph_count=len(histogram_graphs), molecular=molecular_dataset,
                page_index=page_index, total_pages=graphlet_pages,
            )
            if pdf_output:
                _save_drawing(histogram, output, append=total_pages + page_index > 0)
                print(f"Added graphlet page {page_index + 1}/{graphlet_pages} to PDF: {output}", flush=True)
            if histogram_output != output:
                separate_pdf = histogram_output.suffix.lower() == ".pdf"
                page_output = histogram_output if separate_pdf else _page_output_path(histogram_output, page_index, graphlet_pages)
                _save_drawing(histogram, page_output, append=separate_pdf and page_index > 0)
                print(f"Saved graphlet page {page_index + 1}/{graphlet_pages}: {page_output}", flush=True)
            histogram.close()
        histogram_report = (
            output.with_name(f"{output.stem}_graphlet_histogram.json")
            if histogram_output == output
            else histogram_output.with_suffix(".json")
        )
        total_cycle_graphlets = sum(row.count for row in histogram_rows)
        save_json(
            {
                "dataset": dataset_name,
                "split": "all",
                "selected_graphs": len(histogram_graphs),
                "total_dataset_graphs": len(prepared_graphs),
                "valid_molecular_graphs_only": molecular_dataset,
                "excluded_invalid_molecular_graphs": len(prepared_graphs) - len(eligible_indices),
                "molecular_validity_criterion": (
                    "nonempty simple graph; complete atom/bond attributes; RDKit "
                    "sanitization with projected formal-charge inference"
                    if molecular_dataset else None
                ),
                "drawn_split": args.split,
                "drawn_graphs": len(indices),
                "definition": "induced_atom_bond_typed_simple_cycle_Ck" if molecular_dataset else "induced_simple_cycle_Ck",
                **({
                    "node_type": "atomic_number",
                    "edge_type": "bond_type",
                    "edge_type_encoding": {str(number): BOND_LABELS[key] for number, key in BOND_TYPE_KEYS.items()},
                    "canonicalization": "ATTR_CYCLE_V1; rotations and reflections",
                    "type_sequence_order": "edge_types[i] connects node_types[i] to node_types[(i+1) % k]",
                    "observed_types_only": True,
                } if molecular_dataset else {}),
                "normalization": (
                    "count / total induced-cycle count over all requested k"
                ),
                "sort": "frequency_descending_then_k_then_canonical_key" if molecular_dataset else "frequency_descending_then_k_ascending",
                "k_min": args.k_min,
                "k_max": args.k_max,
                "total_eligible_node_subsets": eligible_subsets,
                "total_cycle_graphlets": total_cycle_graphlets,
                "graphlet_pages": graphlet_pages,
                "graphlets": [
                    {
                        "graphlet": row.canonical_key or f"C{row.order}",
                        **({
                            "rank": index + 1,
                            "node_types": list(row.node_types),
                            "edge_types": list(row.edge_types),
                        } if molecular_dataset else {}),
                        "k": row.order,
                        "count": row.count,
                        "frequency": row.frequency,
                        "eligible_node_subsets": row.possible_subsets,
                        "induced_subset_rate": row.subset_rate,
                    }
                    for index, row in enumerate(histogram_rows)
                ],
            },
            histogram_report,
        )
        print(f"Saved graphlet counts: {histogram_report}")

    ok_count = sum(1 for item in loaded_items if item.error is None)
    fail_count = len(loaded_items) - ok_count
    molecular_count = sum(
        1
        for item in loaded_items
        if item.error is None and item.render_mode == "molecule"
    )
    generic_count = sum(
        1
        for item in loaded_items
        if item.error is None and item.render_mode == "generic"
    )
    print(f"Requested graphs: {len(loaded_items)}")
    print(f"Rendered molecular graphs: {molecular_count}")
    print(f"Rendered generic graphs: {generic_count}")
    print(f"Rendered successfully: {ok_count}")
    print(f"Failed to render: {fail_count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
