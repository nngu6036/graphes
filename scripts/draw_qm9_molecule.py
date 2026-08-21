#!/usr/bin/env python3
"""Draw one or more QM9 molecules with coloured atom and bond types.

This version auto-detects the project's QM9 dataset (typically data/QM9)
so the user does not need to pass --root.

Main features
-------------
- auto-detect the QM9 dataset inside the current project
- draw a single molecule or a range of molecules
- arrange molecules in a row x col grid
- if the range is larger than row * col, continue on the next figure/page
- atom colours represent node types; bond colours represent edge types
- works from an existing processed PyG QM9 cache when available, otherwise
  loads directly from raw/gdb9.sdf without preprocessing the whole dataset

Examples
--------
Draw molecule 40 only::

    python scripts/draw_qm9_molecule.py \
      --index-from 40 --index-to 40 \
      --row 1 --col 1 \
      --output outputs/qm9_40.png

Draw molecules 40..55, 4 per row and 3 rows per page::

    python scripts/draw_qm9_molecule.py \
      --index-from 40 --index-to 55 \
      --row 3 --col 4 \
      --output outputs/qm9_40_55.png

If more than 12 molecules are requested in the example above, the script saves
multiple files such as:
- outputs/qm9_40_55_page_001.png
- outputs/qm9_40_55_page_002.png
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class MoleculeInfo:
    source: str
    name: str
    smiles: str
    dataset_index: Optional[int] = None
    raw_index: Optional[int] = None


@dataclass(frozen=True)
class LoadedItem:
    info: MoleculeInfo
    mol: Any | None
    error: str | None = None


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


def _candidate_project_roots() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    cwd = Path.cwd().resolve()
    add(cwd)
    for parent in cwd.parents:
        add(parent)

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        add(parent)

    return candidates


def _detect_qm9_root() -> Path:
    for base in _candidate_project_roots():
        candidate = base / "data" / "QM9"
        if (candidate / "raw" / "gdb9.sdf").is_file() or (candidate / "processed" / "data_v3.pt").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not auto-detect the QM9 dataset. Expected a project dataset like data/QM9/ "
        "with raw/gdb9.sdf or processed/data_v3.pt."
    )


def _find_qm9_raw_file(root: Path, filename: str) -> Optional[Path]:
    for candidate in (root / "raw" / filename, root / filename):
        if candidate.is_file():
            return candidate
    return None


def _read_uncharacterized_indices(path: Optional[Path]) -> set[int]:
    if path is None or not path.is_file():
        return set()

    excluded: set[int] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            one_based = int(parts[0])
        except ValueError:
            continue
        if one_based > 0:
            excluded.add(one_based - 1)
    return excluded


def _pyg_index_to_raw_index(index: int, raw_count: int, excluded: set[int]) -> int:
    if index < 0:
        raise IndexError("index must be non-negative")

    kept_position = 0
    valid_excluded = {i for i in excluded if 0 <= i < raw_count}
    for raw_index in range(raw_count):
        if raw_index in valid_excluded:
            continue
        if kept_position == index:
            return raw_index
        kept_position += 1

    raise IndexError(
        f"PyG QM9 index {index} is outside the filtered dataset after excluding "
        f"{len(valid_excluded)} uncharacterized records"
    )


def _load_from_sdf(
    path: Path,
    index: int,
    *,
    dataset_index: Optional[int] = None,
    source_label: Optional[str] = None,
) -> Tuple[Any, MoleculeInfo]:
    from rdkit import Chem

    if not path.is_file():
        raise FileNotFoundError(f"SDF file does not exist: {path}")
    if index < 0:
        raise IndexError("index must be non-negative")

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    record_count = len(supplier)
    if index >= record_count:
        raise IndexError(f"SDF index {index} is outside [0, {record_count - 1}]")

    mol = supplier[index]
    if mol is None:
        raise ValueError(
            f"RDKit could not read SDF record {index} from {path}. "
            "The record may be malformed or the SDF download may be incomplete."
        )

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"SDF record {index}"
    return mol, MoleculeInfo(
        source=source_label or str(path),
        name=name,
        smiles=_safe_smiles(mol),
        dataset_index=index if dataset_index is None else dataset_index,
        raw_index=index,
    )


def _pyg_graph_to_rdkit(data: Any) -> Any:
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType

    if not hasattr(data, "z") or not hasattr(data, "edge_index"):
        raise ValueError("PyG data object has neither a usable SMILES nor z/edge_index fields")

    z_values = data.z.detach().cpu().tolist()
    edge_index = data.edge_index.detach().cpu()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.detach().cpu()

    rw_mol = Chem.RWMol()
    for atomic_number in z_values:
        rw_mol.AddAtom(Chem.Atom(int(atomic_number)))

    bond_types = {
        0: BondType.SINGLE,
        1: BondType.DOUBLE,
        2: BondType.TRIPLE,
        3: BondType.AROMATIC,
    }

    seen = set()
    for edge_pos in range(edge_index.size(1)):
        u = int(edge_index[0, edge_pos])
        v = int(edge_index[1, edge_pos])
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair in seen:
            continue
        seen.add(pair)

        bond_class = 0
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                bond_class = int(edge_attr[edge_pos])
            else:
                bond_class = int(edge_attr[edge_pos].argmax())
        rw_mol.AddBond(pair[0], pair[1], bond_types.get(bond_class, BondType.SINGLE))

    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        mol.UpdatePropertyCache(strict=False)
    return mol


def _load_from_pyg_qm9(root: Path, index: int) -> Tuple[Any, MoleculeInfo]:
    from rdkit import Chem

    try:
        from torch_geometric.datasets import QM9
    except ImportError as exc:
        raise RuntimeError(
            "PyG input mode requires torch-geometric. Install it or rely on raw/gdb9.sdf."
        ) from exc

    dataset = QM9(root=str(root))
    if index < 0 or index >= len(dataset):
        raise IndexError(f"QM9 index {index} is outside [0, {len(dataset) - 1}]")

    data = dataset[index]
    raw_smiles = _python_scalar(getattr(data, "smiles", None))
    mol = Chem.MolFromSmiles(str(raw_smiles)) if raw_smiles else None
    if mol is None:
        mol = _pyg_graph_to_rdkit(data)

    raw_name = _python_scalar(getattr(data, "name", None), f"QM9[{index}]")
    raw_dataset_idx = _python_scalar(getattr(data, "idx", None), index)
    try:
        raw_dataset_idx = int(raw_dataset_idx)
    except Exception:
        raw_dataset_idx = index

    return mol, MoleculeInfo(
        source=f"PyG QM9 root={root}",
        name=str(raw_name),
        smiles=_safe_smiles(mol),
        dataset_index=index,
        raw_index=raw_dataset_idx,
    )


def _load_from_qm9_raw_root(root: Path, index: int, index_space: str) -> Tuple[Any, MoleculeInfo]:
    from rdkit import Chem

    sdf_path = _find_qm9_raw_file(root, "gdb9.sdf")
    if sdf_path is None:
        raise FileNotFoundError(f"Could not find gdb9.sdf under {root}")

    if index_space == "raw":
        raw_index = index
    elif index_space == "pyg":
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        raw_count = len(supplier)
        excluded = _read_uncharacterized_indices(_find_qm9_raw_file(root, "uncharacterized.txt"))
        raw_index = _pyg_index_to_raw_index(index, raw_count, excluded)
    else:
        raise ValueError(f"Unknown index space: {index_space}")

    return _load_from_sdf(
        sdf_path,
        raw_index,
        dataset_index=index,
        source_label=f"QM9 raw SDF={sdf_path}",
    )


def _load_from_qm9_root(root: Path, index: int, loader: str, index_space: str) -> Tuple[Any, MoleculeInfo]:
    if loader == "raw":
        return _load_from_qm9_raw_root(root, index, index_space)
    if loader == "pyg":
        return _load_from_pyg_qm9(root, index)
    if loader != "auto":
        raise ValueError(f"Unknown loader: {loader}")

    processed_path = root / "processed" / "data_v3.pt"
    raw_sdf = _find_qm9_raw_file(root, "gdb9.sdf")

    if processed_path.is_file():
        try:
            return _load_from_pyg_qm9(root, index)
        except Exception as exc:
            if raw_sdf is None:
                raise
            print(
                f"Warning: processed PyG loading failed ({exc}); falling back to direct raw-SDF access.",
                file=sys.stderr,
            )

    if raw_sdf is not None:
        return _load_from_qm9_raw_root(root, index, index_space)

    return _load_from_pyg_qm9(root, index)


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


def _render_molecule_panel(
    loaded: LoadedItem,
    panel_width: int,
    panel_height: int,
    show_title: bool,
) -> Any:
    from PIL import Image, ImageDraw
    from rdkit.Chem.Draw import rdMolDraw2D

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

    if loaded.error or loaded.mol is None:
        y = 20
        draw.text((16, y), f"QM9 index {loaded.info.dataset_index}", fill="#B42318", font=title_font)
        y += 34
        for line in _wrap_text(loaded.error or "Unknown loading error", 42):
            draw.text((16, y), line, fill="#5F2120", font=body_font)
            y += 20
        return bg

    if show_title:
        draw.text((14, top_pad), f"QM9 index {loaded.info.dataset_index}", fill="#111827", font=title_font)

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

    if loaded.info.raw_index is not None and loaded.info.raw_index != loaded.info.dataset_index:
        raw_text = f"raw {loaded.info.raw_index}"
        bbox = draw.textbbox((0, 0), raw_text, font=small_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x1 = panel_width - tw - 18
        y1 = panel_height - th - 12
        draw.rounded_rectangle((x1 - 8, y1 - 4, x1 + tw + 8, y1 + th + 4), radius=8, fill="#F3F4F6", outline="#D1D5DB")
        draw.text((x1, y1), raw_text, fill="#374151", font=small_font)

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


def _compose_page(
    items: list[LoadedItem],
    row: int,
    col: int,
    panel_width: int,
    panel_height: int,
    page_index: int,
    total_pages: int,
    index_from: int,
    index_to: int,
    qm9_root: Path,
    show_hydrogens: bool,
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

    title = f"QM9 molecules {index_from} to {index_to}"
    if total_pages > 1:
        title += f"  |  page {page_index + 1}/{total_pages}"
    draw.text((outer_pad, outer_pad), title, fill="#111827", font=title_font)
    draw.text((outer_pad, outer_pad + 28), f"Dataset: {qm9_root}", fill="#6B7280", font=body_font)

    start_y = outer_pad + header_h
    for i, item in enumerate(items):
        r = i // col
        c = i % col
        x = outer_pad + c * (panel_width + gap)
        y = start_y + r * (panel_height + gap)
        panel = _render_molecule_panel(item, panel_width, panel_height, show_title=True)
        canvas.paste(panel, (x, y))

    _draw_page_legend(canvas, page_height - legend_h, page_width, show_hydrogens)
    return canvas


def _page_output_path(output: Path, page_index: int, total_pages: int) -> Path:
    if total_pages <= 1:
        return output
    return output.with_name(f"{output.stem}_page_{page_index + 1:03d}{output.suffix}")


def _default_output(index_from: int, index_to: int) -> Path:
    if index_from == index_to:
        return Path(f"outputs/qm9_{index_from:06d}.png")
    return Path(f"outputs/qm9_{index_from:06d}_{index_to:06d}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a range of QM9 molecules with coloured node and edge types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", type=int, help="Backwards-compatible alias for drawing a single molecule.")
    parser.add_argument("--index-from", type=int, default=0, help="First QM9 index to draw (inclusive).")
    parser.add_argument("--index-to", type=int, help="Last QM9 index to draw (inclusive).")
    parser.add_argument("--row", type=int, default=2, help="Number of molecule rows per figure.")
    parser.add_argument("--col", type=int, default=4, help="Number of molecule columns per figure.")
    parser.add_argument("--panel-width", type=int, default=360, help="Width of each molecule panel in pixels.")
    parser.add_argument("--panel-height", type=int, default=300, help="Height of each molecule panel in pixels.")
    parser.add_argument("--output", type=Path, help="Output PNG path. If multiple pages are needed, numbered files are created.")
    parser.add_argument("--loader", choices=("auto", "raw", "pyg"), default="auto")
    parser.add_argument(
        "--index-space",
        choices=("pyg", "raw"),
        default="pyg",
        help="Interpret indices in filtered PyG space or literal raw SDF space when using raw loading.",
    )
    parser.add_argument("--show-hydrogens", action="store_true", help="Draw explicit hydrogens.")
    parser.add_argument("--atom-indices", action="store_true", help="Append RDKit atom indices to atom labels.")
    parser.add_argument("--bond-labels", action="store_true", help="Annotate bonds with single/double/triple/aromatic.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.index is not None:
        index_from = args.index
        index_to = args.index
    else:
        index_from = args.index_from
        index_to = args.index_from if args.index_to is None else args.index_to

    if index_from < 0 or index_to < 0:
        raise ValueError("indices must be non-negative")
    if index_from > index_to:
        raise ValueError("--index-from must be <= --index-to")
    if args.row <= 0 or args.col <= 0:
        raise ValueError("--row and --col must be positive")
    if args.panel_width < 180 or args.panel_height < 180:
        raise ValueError("--panel-width and --panel-height should be at least 180")

    qm9_root = _detect_qm9_root()
    print(f"Using QM9 dataset: {qm9_root}")

    output = (args.output or _default_output(index_from, index_to)).expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("This grid script currently writes PNG output only; please use a .png output path.")
    output.parent.mkdir(parents=True, exist_ok=True)

    indices = list(range(index_from, index_to + 1))
    loaded_items: list[LoadedItem] = []
    for index in indices:
        try:
            mol, info = _load_from_qm9_root(qm9_root, index, args.loader, args.index_space)
            mol = _prepare_molecule(
                mol,
                show_hydrogens=args.show_hydrogens,
                atom_indices=args.atom_indices,
                bond_labels=args.bond_labels,
            )
            loaded_items.append(LoadedItem(info=info, mol=mol))
        except Exception as exc:
            loaded_items.append(
                LoadedItem(
                    info=MoleculeInfo(
                        source=f"QM9 root={qm9_root}",
                        name=f"QM9[{index}]",
                        smiles="<unavailable>",
                        dataset_index=index,
                        raw_index=None,
                    ),
                    mol=None,
                    error=str(exc),
                )
            )

    per_page = args.row * args.col
    total_pages = math.ceil(len(loaded_items) / per_page)
    saved_paths: list[Path] = []
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
            index_from=index_from,
            index_to=index_to,
            qm9_root=qm9_root,
            show_hydrogens=args.show_hydrogens,
        )
        page_output = _page_output_path(output, page_index, total_pages)
        canvas.save(page_output)
        saved_paths.append(page_output)
        print(f"Saved: {page_output}")

    ok_count = sum(1 for item in loaded_items if item.error is None)
    fail_count = len(loaded_items) - ok_count
    print(f"Requested molecules: {len(loaded_items)}")
    print(f"Loaded successfully: {ok_count}")
    print(f"Failed to load: {fail_count}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
