#!/usr/bin/env python3
"""Draw a QM9 molecule with colours for atom/node and bond/edge types.

The default input is a molecule from ``torch_geometric.datasets.QM9``.
The script can also draw a record from a raw SDF file or a direct SMILES
string, which is useful for visualising generated GraphER samples.

Examples
--------
Draw molecule 42 from PyG QM9::

    python draw_qm9_molecule.py \
        --root data/QM9 \
        --index 42 \
        --output outputs/qm9_42.svg

Draw molecule 42 directly from the raw QM9 SDF::

    python draw_qm9_molecule.py \
        --sdf data/QM9/raw/gdb9.sdf \
        --index 42 \
        --output outputs/qm9_42.png

Draw a generated molecule from SMILES::

    python draw_qm9_molecule.py \
        --smiles 'N#CC(=O)F' \
        --output outputs/generated.svg \
        --atom-indices --bond-labels

Dependencies
------------
- rdkit
- Pillow (only needed for PNG output)
- torch-geometric (only needed for the default PyG-QM9 input mode)
"""

from __future__ import annotations

import argparse
import io
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


# Conventional, high-contrast colours. Unknown types use the fallback entry.
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

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass(frozen=True)
class MoleculeInfo:
    source: str
    name: str
    smiles: str
    dataset_index: Optional[int] = None


def _hex_to_rgb255(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex colour, received {value!r}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _hex_to_rgb01(value: str) -> Tuple[float, float, float]:
    return tuple(channel / 255.0 for channel in _hex_to_rgb255(value))  # type: ignore[return-value]


def _python_scalar(value: Any, default: Any = None) -> Any:
    """Convert common tensor/list wrappers into a Python scalar."""
    if value is None:
        return default
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _python_scalar(value[0], default)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def _safe_smiles(mol: Any) -> str:
    from rdkit import Chem

    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return "<unavailable>"


def _load_from_smiles(smiles: str) -> Tuple[Any, MoleculeInfo]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the supplied SMILES: {smiles!r}")
    return mol, MoleculeInfo(source="SMILES", name="generated molecule", smiles=_safe_smiles(mol))


def _load_from_sdf(path: Path, index: int) -> Tuple[Any, MoleculeInfo]:
    from rdkit import Chem

    if not path.is_file():
        raise FileNotFoundError(f"SDF file does not exist: {path}")
    if index < 0:
        raise IndexError("--index must be non-negative")

    # sanitize=False allows the record to be read even when a toolkit version
    # applies stricter chemistry rules. We sanitize explicitly below and retain
    # a drawable molecule when only non-critical sanitization steps fail.
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    if index >= len(supplier):
        raise IndexError(f"SDF index {index} is outside [0, {len(supplier) - 1}]")
    mol = supplier[index]
    if mol is None:
        raise ValueError(f"RDKit could not read record {index} from {path}")

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        print(
            f"Warning: full RDKit sanitization failed for SDF record {index}: {exc}. "
            "Drawing the parsed molecular graph.",
            file=sys.stderr,
        )
        mol.UpdatePropertyCache(strict=False)

    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"SDF record {index}"
    return mol, MoleculeInfo(
        source=str(path),
        name=name,
        smiles=_safe_smiles(mol),
        dataset_index=index,
    )


def _pyg_graph_to_rdkit(data: Any) -> Any:
    """Fallback conversion for old PyG QM9 objects without ``data.smiles``."""
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
        bond_type = bond_types.get(bond_class, BondType.SINGLE)
        rw_mol.AddBond(pair[0], pair[1], bond_type)
        if bond_type == BondType.AROMATIC:
            bond = rw_mol.GetBondBetweenAtoms(pair[0], pair[1])
            bond.SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(pair[0]).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(pair[1]).SetIsAromatic(True)

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
            "PyG input mode requires torch-geometric. Install it, or use "
            "--sdf /path/to/gdb9.sdf or --smiles instead."
        ) from exc

    if index < 0:
        raise IndexError("--index must be non-negative")

    dataset = QM9(root=str(root))
    if index >= len(dataset):
        raise IndexError(f"QM9 index {index} is outside [0, {len(dataset) - 1}]")

    data = dataset[index]
    raw_smiles = _python_scalar(getattr(data, "smiles", None))
    mol = Chem.MolFromSmiles(str(raw_smiles)) if raw_smiles else None
    if mol is None:
        mol = _pyg_graph_to_rdkit(data)

    raw_name = _python_scalar(getattr(data, "name", None), f"QM9[{index}]")
    raw_dataset_idx = _python_scalar(getattr(data, "idx", None), index)
    try:
        dataset_idx = int(raw_dataset_idx)
    except (TypeError, ValueError):
        dataset_idx = index

    return mol, MoleculeInfo(
        source=f"PyG QM9 root={root}",
        name=str(raw_name),
        smiles=_safe_smiles(mol),
        dataset_index=dataset_idx,
    )


def _prepare_molecule(mol: Any, show_hydrogens: bool, atom_indices: bool, bond_labels: bool) -> Any:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    mol = Chem.Mol(mol)
    if show_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        # QM9 is commonly visualised as a heavy-atom graph with implicit H.
        try:
            mol = Chem.RemoveHs(mol)
        except Exception:
            mol = Chem.RemoveHs(mol, sanitize=False)
            mol.UpdatePropertyCache(strict=False)

    if mol.GetNumAtoms() == 0:
        raise ValueError("The molecule has no drawable atoms after hydrogen handling")

    rdDepictor.Compute2DCoords(mol, canonOrient=True)

    # Force all node types to be visible, including carbon atoms, which RDKit
    # normally omits in skeletal depictions.
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom.SetProp("atomLabel", f"{symbol}{atom.GetIdx()}" if atom_indices else symbol)

    if bond_labels:
        for bond in mol.GetBonds():
            key = str(bond.GetBondType()).upper()
            bond.SetProp("bondNote", BOND_LABELS.get(key, "other"))

    return mol


def _highlight_maps(mol: Any) -> Tuple[list[int], dict[int, Tuple[float, float, float]], dict[int, float], list[int], dict[int, Tuple[float, float, float]]]:
    atoms = []
    atom_colours = {}
    atom_radii = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        colour = ATOM_COLOURS_HEX.get(symbol, ATOM_COLOURS_HEX["OTHER"])
        atoms.append(idx)
        atom_colours[idx] = _hex_to_rgb01(colour)
        atom_radii[idx] = 0.34 if symbol != "H" else 0.25

    bonds = []
    bond_colours = {}
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        key = str(bond.GetBondType()).upper()
        colour = BOND_COLOURS_HEX.get(key, BOND_COLOURS_HEX["OTHER"])
        bonds.append(idx)
        bond_colours[idx] = _hex_to_rgb01(colour)

    return atoms, atom_colours, atom_radii, bonds, bond_colours


def _configure_drawer(drawer: Any) -> None:
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()  # text/chemical lines stay neutral; type is encoded by highlight colour
    opts.fillHighlights = True
    opts.atomHighlightsAreCircles = True
    opts.continuousHighlight = False
    opts.highlightBondWidthMultiplier = 16
    opts.bondLineWidth = 2.2
    opts.annotationFontScale = 0.58
    opts.padding = 0.08
    opts.addAtomIndices = False  # indices are embedded into atomLabel when requested


def _svg_tag(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def _svg_text(parent: Any, x: int, y: int, text: str, size: int = 18, weight: str = "normal") -> None:
    element = ET.SubElement(
        parent,
        _svg_tag("text"),
        {
            "x": str(x),
            "y": str(y),
            "font-family": "sans-serif",
            "font-size": str(size),
            "font-weight": weight,
            "fill": "#212529",
        },
    )
    element.text = text


def _append_svg_legend(
    svg: str,
    molecule_width: int,
    total_width: int,
    height: int,
    info: MoleculeInfo,
    show_hydrogens: bool,
) -> str:
    root = ET.fromstring(svg)
    root.set("width", f"{total_width}px")
    root.set("height", f"{height}px")
    root.set("viewBox", f"0 0 {total_width} {height}")

    group = ET.SubElement(root, _svg_tag("g"), {"id": "type-legend"})
    ET.SubElement(
        group,
        _svg_tag("rect"),
        {
            "x": str(molecule_width),
            "y": "0",
            "width": str(total_width - molecule_width),
            "height": str(height),
            "fill": "#FFFFFF",
        },
    )
    ET.SubElement(
        group,
        _svg_tag("line"),
        {
            "x1": str(molecule_width),
            "y1": "24",
            "x2": str(molecule_width),
            "y2": str(height - 24),
            "stroke": "#CED4DA",
            "stroke-width": "2",
        },
    )

    x0 = molecule_width + 28
    y = 44
    _svg_text(group, x0, y, "Molecule", size=24, weight="bold")
    y += 30
    _svg_text(group, x0, y, info.name, size=17, weight="bold")
    y += 24
    index_text = "n/a" if info.dataset_index is None else str(info.dataset_index)
    _svg_text(group, x0, y, f"Index: {index_text}", size=15)
    y += 22
    _svg_text(group, x0, y, f"H atoms: {'explicit' if show_hydrogens else 'implicit'}", size=15)
    y += 22
    smiles_text = info.smiles if len(info.smiles) <= 35 else info.smiles[:32] + "..."
    _svg_text(group, x0, y, f"SMILES: {smiles_text}", size=14)

    y += 44
    _svg_text(group, x0, y, "Atom / node type", size=21, weight="bold")
    y += 30
    atom_keys = ["H", "C", "N", "O", "F"] if show_hydrogens else ["C", "N", "O", "F"]
    for symbol in atom_keys:
        ET.SubElement(
            group,
            _svg_tag("circle"),
            {
                "cx": str(x0 + 11),
                "cy": str(y - 6),
                "r": "10",
                "fill": ATOM_COLOURS_HEX[symbol],
                "stroke": "#495057",
                "stroke-width": "1.2",
            },
        )
        _svg_text(group, x0 + 34, y, symbol, size=17)
        y += 29

    y += 18
    _svg_text(group, x0, y, "Bond / edge type", size=21, weight="bold")
    y += 31
    for key in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"):
        colour = BOND_COLOURS_HEX[key]
        line_count = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1}[key]
        offsets = {1: [0], 2: [-3, 3], 3: [-5, 0, 5]}[line_count]
        for offset in offsets:
            attrs = {
                "x1": str(x0),
                "y1": str(y - 7 + offset),
                "x2": str(x0 + 42),
                "y2": str(y - 7 + offset),
                "stroke": colour,
                "stroke-width": "4",
                "stroke-linecap": "round",
            }
            if key == "AROMATIC":
                attrs["stroke-dasharray"] = "7 5"
            ET.SubElement(group, _svg_tag("line"), attrs)
        _svg_text(group, x0 + 56, y, BOND_LABELS[key], size=17)
        y += 31

    return ET.tostring(root, encoding="unicode")


def _draw_svg(
    mol: Any,
    output: Path,
    info: MoleculeInfo,
    width: int,
    height: int,
    legend_width: int,
    show_legend: bool,
    show_hydrogens: bool,
    title: str,
) -> None:
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule_width = width - legend_width if show_legend else width
    if molecule_width < 300:
        raise ValueError("Drawing area is too narrow; increase --width or reduce --legend-width")

    atoms, atom_colours, atom_radii, bonds, bond_colours = _highlight_maps(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(molecule_width, height)
    _configure_drawer(drawer)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=title,
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors=atom_colours,
        highlightBondColors=bond_colours,
        highlightAtomRadii=atom_radii,
        kekulize=False,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    if show_legend:
        svg = _append_svg_legend(svg, molecule_width, width, height, info, show_hydrogens)
    output.write_text(svg, encoding="utf-8")


def _load_font(size: int, bold: bool = False) -> Any:
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


def _draw_png_legend(
    canvas: Any,
    molecule_width: int,
    width: int,
    height: int,
    info: MoleculeInfo,
    show_hydrogens: bool,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((molecule_width, 0, width, height), fill="white")
    draw.line((molecule_width, 24, molecule_width, height - 24), fill="#CED4DA", width=2)

    title_font = _load_font(24, bold=True)
    section_font = _load_font(20, bold=True)
    body_font = _load_font(16)
    body_bold = _load_font(17, bold=True)

    x0 = molecule_width + 28
    y = 30
    draw.text((x0, y), "Molecule", fill="#212529", font=title_font)
    y += 38
    draw.text((x0, y), info.name, fill="#212529", font=body_bold)
    y += 27
    index_text = "n/a" if info.dataset_index is None else str(info.dataset_index)
    draw.text((x0, y), f"Index: {index_text}", fill="#212529", font=body_font)
    y += 23
    draw.text(
        (x0, y),
        f"H atoms: {'explicit' if show_hydrogens else 'implicit'}",
        fill="#212529",
        font=body_font,
    )
    y += 23
    smiles_text = info.smiles if len(info.smiles) <= 35 else info.smiles[:32] + "..."
    draw.text((x0, y), f"SMILES: {smiles_text}", fill="#212529", font=_load_font(14))

    y += 43
    draw.text((x0, y), "Atom / node type", fill="#212529", font=section_font)
    y += 32
    atom_keys = ["H", "C", "N", "O", "F"] if show_hydrogens else ["C", "N", "O", "F"]
    for symbol in atom_keys:
        draw.ellipse((x0, y, x0 + 20, y + 20), fill=ATOM_COLOURS_HEX[symbol], outline="#495057", width=1)
        draw.text((x0 + 34, y - 1), symbol, fill="#212529", font=body_font)
        y += 29

    y += 14
    draw.text((x0, y), "Bond / edge type", fill="#212529", font=section_font)
    y += 34
    for key in ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"):
        colour = BOND_COLOURS_HEX[key]
        line_count = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1}[key]
        offsets = {1: [0], 2: [-3, 3], 3: [-5, 0, 5]}[line_count]
        for offset in offsets:
            if key == "AROMATIC":
                for start in range(x0, x0 + 43, 12):
                    draw.line((start, y + 9 + offset, min(start + 7, x0 + 42), y + 9 + offset), fill=colour, width=4)
            else:
                draw.line((x0, y + 9 + offset, x0 + 42, y + 9 + offset), fill=colour, width=4)
        draw.text((x0 + 56, y), BOND_LABELS[key], fill="#212529", font=body_font)
        y += 31


def _draw_png(
    mol: Any,
    output: Path,
    info: MoleculeInfo,
    width: int,
    height: int,
    legend_width: int,
    show_legend: bool,
    show_hydrogens: bool,
    title: str,
) -> None:
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    molecule_width = width - legend_width if show_legend else width
    if molecule_width < 300:
        raise ValueError("Drawing area is too narrow; increase --width or reduce --legend-width")

    atoms, atom_colours, atom_radii, bonds, bond_colours = _highlight_maps(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(molecule_width, height)
    _configure_drawer(drawer)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=title,
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors=atom_colours,
        highlightBondColors=bond_colours,
        highlightAtomRadii=atom_radii,
        kekulize=False,
    )
    drawer.FinishDrawing()

    molecule_image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    if show_legend:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(molecule_image, (0, 0))
        _draw_png_legend(canvas, molecule_width, width, height, info, show_hydrogens)
    else:
        canvas = molecule_image
    canvas.save(output)


def _default_output(args: argparse.Namespace) -> Path:
    if args.smiles:
        return Path("molecule_coloured.svg")
    return Path(f"qm9_{args.index:06d}_coloured.svg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw a QM9 molecule with atom/node and bond/edge type colours.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--smiles",
        type=str,
        help="Draw this SMILES instead of loading the QM9 dataset.",
    )
    input_group.add_argument(
        "--sdf",
        type=Path,
        help="Draw --index from this SDF file, e.g. QM9 raw/gdb9.sdf.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/QM9"),
        help="Root directory for torch_geometric.datasets.QM9.",
    )
    parser.add_argument("--index", type=int, default=0, help="QM9/PyG or SDF record index.")
    parser.add_argument("--output", type=Path, help="Output .svg or .png path.")
    parser.add_argument("--width", type=int, default=1000, help="Total image width in pixels.")
    parser.add_argument("--height", type=int, default=620, help="Image height in pixels.")
    parser.add_argument(
        "--legend-width",
        type=int,
        default=310,
        help="Width reserved for the colour legend.",
    )
    parser.add_argument(
        "--show-hydrogens",
        action="store_true",
        help="Draw explicit hydrogens. By default, hydrogens are implicit.",
    )
    parser.add_argument(
        "--atom-indices",
        action="store_true",
        help="Append the RDKit atom index to every atom label.",
    )
    parser.add_argument(
        "--bond-labels",
        action="store_true",
        help="Annotate every bond with single/double/triple/aromatic.",
    )
    parser.add_argument("--no-legend", action="store_true", help="Do not draw the type legend.")
    parser.add_argument("--title", type=str, help="Custom text below the molecule.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.legend_width < 0:
        raise ValueError("--legend-width must be non-negative")

    if args.smiles:
        mol, info = _load_from_smiles(args.smiles)
    elif args.sdf:
        mol, info = _load_from_sdf(args.sdf, args.index)
    else:
        mol, info = _load_from_pyg_qm9(args.root, args.index)

    mol = _prepare_molecule(
        mol,
        show_hydrogens=args.show_hydrogens,
        atom_indices=args.atom_indices,
        bond_labels=args.bond_labels,
    )

    output = args.output or _default_output(args)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix not in {".svg", ".png"}:
        raise ValueError("--output must end with .svg or .png")

    title = args.title or (
        f"{info.name} | QM9 index {info.dataset_index}"
        if info.dataset_index is not None
        else info.name
    )
    show_legend = not args.no_legend

    if suffix == ".svg":
        _draw_svg(
            mol,
            output,
            info,
            width=args.width,
            height=args.height,
            legend_width=args.legend_width,
            show_legend=show_legend,
            show_hydrogens=args.show_hydrogens,
            title=title,
        )
    else:
        _draw_png(
            mol,
            output,
            info,
            width=args.width,
            height=args.height,
            legend_width=args.legend_width,
            show_legend=show_legend,
            show_hydrogens=args.show_hydrogens,
            title=title,
        )

    print(f"Saved: {output}")
    print(f"Source: {info.source}")
    print(f"Name: {info.name}")
    print(f"SMILES: {info.smiles}")
    print(f"Atoms shown: {mol.GetNumAtoms()}; bonds shown: {mol.GetNumBonds()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
