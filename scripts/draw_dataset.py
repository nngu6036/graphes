#!/usr/bin/env python3
"""Draw molecular or generic graphs from a prepared dataset by name.

The input is a prepared molecular dataset under
``outputs/datasets/<dataset>``. Pass its name with ``--dataset``; the script
then resolves the directory using the same convention as the training and
evaluation commands. ``--count`` graphs are sampled without replacement;
``--seed`` makes the selection reproducible. Use ``--all`` to draw every graph
in the selected split, or combine it with ``--split all`` for the full dataset.

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

When both ``--k-min`` and ``--k-max`` are supplied, the script also writes a
descending histogram of induced simple-cycle graphlets. For example, the
command above creates ``outputs/community_small_all_graphlet_histogram.png``
and a JSON sidecar containing the raw counts and normalization details.
"""

from __future__ import annotations

import argparse
import io
import itertools
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


def _load_from_prepared_graph(
    graph: Any,
    index: int,
    dataset_name: str,
    split: str,
) -> Tuple[Any, MoleculeInfo]:
    """Convert one prepared attributed NetworkX graph into an RDKit molecule."""

    from rdkit import Chem

    if not isinstance(graph, nx.Graph):
        raise TypeError(
            f"{dataset_name}/{split}[{index}] is not a NetworkX graph "
            f"({type(graph).__name__})."
        )

    normalized = nx.Graph()
    node_map: dict[Any, int] = {}
    for normalized_index, (node, data) in enumerate(graph.nodes(data=True)):
        atomic_number = _first_attribute(
            data,
            ("atomic_num", "atomic_number", "atom_type", "z"),
        )
        if atomic_number is None:
            raise ValueError(
                f"Node {node!r} in {dataset_name}/{split}[{index}] is missing "
                "atomic_num/atom_type. Use an attributed molecular dataset "
                "such as 'qm9_attributed', not a topology-only dataset."
            )
        try:
            atomic_number = int(_python_scalar(atomic_number))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Node {node!r} in {dataset_name}/{split}[{index}] has invalid "
                f"atomic number {atomic_number!r}."
            ) from exc
        node_map[node] = normalized_index
        normalized.add_node(normalized_index, atomic_num=atomic_number)

    for u, v, data in graph.edges(data=True):
        raw_bond_type = _first_attribute(
            data,
            ("bond_type", "edge_type", "bond_order"),
        )
        if raw_bond_type is None:
            raise ValueError(
                f"Edge {(u, v)!r} in {dataset_name}/{split}[{index}] is missing "
                "bond_type. Use an attributed molecular dataset such as "
                "'qm9_attributed', not a topology-only dataset."
            )
        normalized.add_edge(
            node_map[u],
            node_map[v],
            bond_type=_prepared_bond_type(raw_bond_type, edge=(u, v)),
        )

    mol = nx_to_rdkit_mol(
        normalized,
        sanitize=False,
        infer_projected_formal_charges=True,
    )
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)

    index_label = f"{dataset_name}/{split}[{index}]"
    name = str(graph.graph.get("name", index_label))
    return mol, MoleculeInfo(
        source=f"prepared dataset {dataset_name}/{split}",
        name=name,
        smiles=_safe_smiles(mol),
        dataset_index=index,
        source_index=_prepared_source_index(graph),
        index_label=index_label,
    )


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
) -> list[CycleGraphletFrequency]:
    """Count induced cycle graphlets and sort their frequencies descending."""

    if k_min < 3:
        raise ValueError("Cycle graphlets require --graphlet-k-min >= 3.")
    if k_max < k_min:
        raise ValueError("--graphlet-k-max must be >= --graphlet-k-min.")
    if not all(isinstance(graph, nx.Graph) for graph in graphs):
        raise TypeError("Graphlet histograms require NetworkX graphs.")

    counts = {order: 0 for order in range(k_min, k_max + 1)}
    possible = {order: 0 for order in range(k_min, k_max + 1)}
    for graph in graphs:
        simple_graph = nx.Graph(graph)
        nodes = tuple(simple_graph.nodes())
        adjacency = {
            node: set(simple_graph.neighbors(node)) for node in nodes
        }
        for order in counts:
            if len(nodes) < order:
                continue
            possible[order] += math.comb(len(nodes), order)
            counts[order] += sum(
                1
                for subset in itertools.combinations(nodes, order)
                if _is_induced_cycle_from_adjacency(adjacency, subset)
            )

    total_cycles = sum(counts.values())
    rows = [
        CycleGraphletFrequency(
            order=order,
            count=counts[order],
            possible_subsets=possible[order],
            frequency=(counts[order] / total_cycles if total_cycles else 0.0),
            subset_rate=(
                counts[order] / possible[order] if possible[order] else 0.0
            ),
        )
        for order in counts
    ]
    return sorted(rows, key=lambda row: (-row.frequency, row.order))


def _render_cycle_graphlet_histogram(
    rows: Sequence[CycleGraphletFrequency],
    *,
    dataset_label: str,
    graph_count: int,
) -> Any:
    """Render a descending horizontal histogram of induced cycle graphlets."""

    from PIL import Image, ImageDraw

    width = 1100
    row_height = 58
    top = 112
    bottom = 68
    height = max(320, top + bottom + row_height * max(len(rows), 1))
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(24, bold=True)
    body_font = _load_font(15)
    label_font = _load_font(16, bold=True)
    small_font = _load_font(13)

    draw.text(
        (28, 22),
        f"{dataset_label}: induced cycle graphlet histogram",
        fill="#111827",
        font=title_font,
    )
    draw.text(
        (28, 58),
        (
            f"graphs={graph_count} | only chordless induced subgraphs Ck | "
            "bars sorted by cycle-share frequency"
        ),
        fill="#4B5563",
        font=body_font,
    )

    label_x = 34
    bar_x = 118
    bar_width = 580
    value_x = bar_x + bar_width + 18
    maximum = max((row.frequency for row in rows), default=0.0)
    scale = maximum if maximum > 0.0 else 1.0
    colours = ("#2563EB", "#0891B2", "#16A34A", "#9333EA", "#EA580C")

    for index, row in enumerate(rows):
        y = top + index * row_height
        draw.text((label_x, y + 8), f"C{row.order}", fill="#111827", font=label_font)
        draw.rounded_rectangle(
            (bar_x, y + 6, bar_x + bar_width, y + 34),
            radius=8,
            fill="#EEF2F7",
        )
        filled = int(round(bar_width * row.frequency / scale))
        if filled > 0:
            draw.rounded_rectangle(
                (bar_x, y + 6, bar_x + max(filled, 3), y + 34),
                radius=8,
                fill=colours[index % len(colours)],
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
        draw.text((28, top), "No graphlet orders requested.", fill="#6B7280", font=body_font)
    draw.line((28, height - 44, width - 28, height - 44), fill="#D1D5DB", width=2)
    draw.text(
        (28, height - 32),
        "frequency = Ck count / total induced-cycle count over all requested k",
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
    return graph_output.with_name(f"{graph_output.stem}_graphlet_histogram.png")


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
        help="Number of distinct graphs to sample without replacement.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Draw every graph in the selected split or splits.",
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
            "Output PNG path. If multiple pages are needed, numbered files "
            "are created."
        ),
    )
    parser.add_argument(
        "--k-min",
        "--graphlet-k-min",
        dest="k_min",
        type=int,
        help=(
            "Minimum induced cycle-graphlet order. Must be provided together "
            "with --k-max and be at least 3."
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
            "Optional PNG path for the induced-cycle histogram. By default, "
            "_graphlet_histogram is appended to the --output stem."
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

    print(
        f"Resolving prepared dataset {args.dataset!r} under {args.root}...",
        flush=True,
    )
    prepared_graphs, dataset_path, dataset_name, graph_locations = (
        _load_prepared_dataset_selection(
            args.dataset,
            args.root,
            args.split,
        )
    )
    indices = _select_graph_indices(
        len(prepared_graphs),
        count=args.count,
        seed=args.seed,
        draw_all=args.all,
    )
    dataset_label = f"{dataset_name}/{args.split}"
    output_prefix = f"{dataset_name}_{args.split}"
    print(
        f"Using prepared dataset: {dataset_name} split={args.split} "
        f"graphs={len(prepared_graphs)} path={dataset_path}",
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
    if output.suffix.lower() != ".png":
        raise ValueError(
            "This grid script currently writes PNG output only; please use a "
            ".png output path."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    loaded_items: list[LoadedItem] = []
    for index in indices:
        selected_split, split_index = graph_locations[index]
        try:
            graph = prepared_graphs[index]
            if _is_molecular_graph(graph):
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
        page_output = _page_output_path(output, page_index, total_pages)
        canvas.save(page_output)
        print(f"Saved: {page_output}")

    if graphlet_requested:
        selected_graphs = [prepared_graphs[index] for index in indices]
        eligible_subsets = sum(
            math.comb(graph.number_of_nodes(), order)
            for graph in selected_graphs
            for order in range(args.k_min, args.k_max + 1)
            if isinstance(graph, nx.Graph)
            and graph.number_of_nodes() >= order
        )
        print(
            "Counting exact induced cycle graphlets "
            f"C{args.k_min}..C{args.k_max} over "
            f"{eligible_subsets:,} eligible node subsets...",
            flush=True,
        )
        histogram_rows = _cycle_graphlet_histogram(
            selected_graphs,
            k_min=args.k_min,
            k_max=args.k_max,
        )
        histogram_output = _graphlet_histogram_output_path(
            output,
            args.graphlet_output,
        )
        if histogram_output.suffix.lower() != ".png":
            raise ValueError("--graphlet-output must use a .png extension")
        histogram_output.parent.mkdir(parents=True, exist_ok=True)
        histogram = _render_cycle_graphlet_histogram(
            histogram_rows,
            dataset_label=dataset_label,
            graph_count=len(selected_graphs),
        )
        histogram.save(histogram_output)
        histogram_report = histogram_output.with_suffix(".json")
        total_cycle_graphlets = sum(row.count for row in histogram_rows)
        save_json(
            {
                "dataset": dataset_name,
                "split": args.split,
                "selected_graphs": len(selected_graphs),
                "definition": "induced_simple_cycle_Ck",
                "normalization": (
                    "count / total induced-cycle count over all requested k"
                ),
                "sort": "frequency_descending_then_k_ascending",
                "k_min": args.k_min,
                "k_max": args.k_max,
                "total_eligible_node_subsets": eligible_subsets,
                "total_cycle_graphlets": total_cycle_graphlets,
                "graphlets": [
                    {
                        "graphlet": f"C{row.order}",
                        "k": row.order,
                        "count": row.count,
                        "frequency": row.frequency,
                        "eligible_node_subsets": row.possible_subsets,
                        "induced_subset_rate": row.subset_rate,
                    }
                    for row in histogram_rows
                ],
            },
            histogram_report,
        )
        print(f"Saved graphlet histogram: {histogram_output}")
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
