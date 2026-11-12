#!/usr/bin/env python3
"""Draw one colour PNG grid from generated QM9 or ZINC molecular graphs.

Example (4 rows, 6 columns):
    python scripts/plot_molecular_dataset.py --dataset qm9 --row 4 --column 6 \
        --generated-path /path/to/generation/ablations/TR --seed 42

Place this script in GraphES/scripts/. Input must be a trusted project pickle:
a list/tuple of attributed NetworkX graphs, or a dictionary wrapping that list.
Atom attributes: atomic_num (or atom_type) and optional formal_charge.
Bond categories: bond_type=1/2/3/4 for single/double/triple/aromatic.

Samples uniformly without replacement among raw RDKit-valid records. Duplicate
molecules remain distinct records; no bond correction or charge inference is
applied. Exclusions and original sample indices are recorded in a JSON sidecar.
Disconnected valid structures are retained. Colours denote elements.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import random
import sys
from pathlib import Path

import networkx as nx
from PIL import Image

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError as error:
    raise SystemExit("RDKit is required. Run this script in your molecular GraphES environment.") from error

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from grapher.rewiring_mlp.molecular.graph_io import nx_to_rdkit_mol
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def resolve_generated_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        for filename in ("base_graphs.pkl", "molecular_graphs.pkl", "generated_graphs.pkl"):
            candidate = path / filename
            if candidate.is_file():
                return candidate
        raise ValueError(f"No base_graphs.pkl, molecular_graphs.pkl or generated_graphs.pkl in {path}")
    if not path.is_file():
        raise ValueError(f"Generated file does not exist: {path}")
    return path


def load_records(path: Path) -> list:
    data = load_trusted_networkx_pickle(path)
    if isinstance(data, dict):
        for key in ("graphs", "molecular_graphs", "generated_graphs", "base_graphs"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("Expected a nonempty list/tuple of attributed NetworkX graphs, optionally wrapped in a dictionary.")
    return list(data)


def integer_attribute(value, name: str) -> int:
    numeric = float(value)
    if isinstance(value, bool) or not math.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(numeric)


def raw_molecule(graph):
    if not isinstance(graph, nx.Graph) or not graph or graph.is_directed() or graph.is_multigraph():
        raise ValueError("Expected a nonempty simple undirected molecular graph")
    if nx.number_of_selfloops(graph):
        raise ValueError("Molecular graph contains self-loops")
    # The existing converter expects integer labels. Relabel only a drawing copy.
    normalized = nx.convert_node_labels_to_integers(graph, ordering="default")
    for node, attrs in normalized.nodes(data=True):
        atomic_num = integer_attribute(attrs.get("atomic_num", attrs.get("atom_type")), f"Atom {node}: atomic_num")
        if not 1 <= atomic_num <= 118:
            raise ValueError(f"Atom {node}: atomic_num must be between 1 and 118")
        attrs["atomic_num"] = atomic_num
        if attrs.get("formal_charge") is not None:
            attrs["formal_charge"] = integer_attribute(attrs["formal_charge"], f"Atom {node}: formal_charge")
    for u, v, attrs in normalized.edges(data=True):
        bond_type = integer_attribute(attrs.get("bond_type"), f"Bond {u, v}: bond_type")
        if bond_type not in (1, 2, 3, 4):
            raise ValueError(f"Bond {u, v}: bond_type must be 1, 2, 3 or 4")
        attrs["bond_type"] = bond_type
    mol = nx_to_rdkit_mol(normalized, sanitize=True, infer_projected_formal_charges=False)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return mol, smiles


def draw_panel(mol, args, index: int) -> Image.Image:
    mol = Chem.Mol(mol)
    mol = Chem.AddHs(mol) if args.show_hydrogens else Chem.RemoveHs(mol)
    rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(args.panel_width, args.panel_height)
    options = drawer.drawOptions()
    options.setBackgroundColour((1.0, 1.0, 1.0))
    options.padding = 0.06
    options.bondLineWidth = 2.0
    options.minFontSize = 20
    options.maxFontSize = 32
    options.legendFontSize = 16
    options.updateAtomPalette({
        1: (0.18, 0.18, 0.18), 6: (0.18, 0.18, 0.18),
        7: (0.20, 0.28, 0.72), 8: (0.95, 0.12, 0.12),
        9: (0.00, 0.65, 0.75), 15: (0.90, 0.45, 0.10),
        16: (0.72, 0.57, 0.02), 17: (0.10, 0.60, 0.20),
        35: (0.60, 0.20, 0.10), 53: (0.52, 0.20, 0.65),
    })
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, legend=f"Graph {index}" if args.show_indices else "")
    drawer.FinishDrawing()
    with Image.open(io.BytesIO(drawer.GetDrawingText())) as image:
        return image.convert("RGB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("qm9", "zinc"))
    parser.add_argument("--row", "--rows", dest="rows", required=True, type=positive_int)
    parser.add_argument("--column", "--col", "--cols", dest="cols", required=True, type=positive_int)
    parser.add_argument("--generated-path", "--generated-graphs", "--generated-dir", dest="generated_path",
                        required=True, type=Path, help="Attributed graph pickle, or its containing directory.")
    parser.add_argument("--output-dir", type=Path, help="Default: plots/ beside the selected pickle.")
    parser.add_argument("--output", type=Path, help="Explicit output PNG path; overrides --output-dir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-width", type=positive_int, default=420, help="Pixels per molecule, horizontally.")
    parser.add_argument("--panel-height", type=positive_int, default=320, help="Pixels per molecule, vertically.")
    parser.add_argument("--dpi", type=positive_int, default=300, help="PNG print-resolution metadata.")
    parser.add_argument("--show-hydrogens", action="store_true", help="Draw all hydrogen atoms explicitly.")
    parser.add_argument("--show-indices", action="store_true", help="Label panels with original file indices.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if min(args.panel_width, args.panel_height) < 120:
            raise ValueError("Panel dimensions must be at least 120 pixels")
        path = resolve_generated_file(args.generated_path)
        records = load_records(path)
        count = args.rows * args.cols
        output_dir = (args.output_dir or path.parent / "plots").expanduser().resolve()
        output = (args.output or output_dir / f"{args.dataset}_generated_{args.rows}x{args.cols}.png").expanduser().resolve()
        if output.suffix.lower() != ".png":
            raise ValueError("--output must end in .png")
        print(f"Checking {len(records):,} generated molecular graphs from {path}...", flush=True)
        valid_indices, exclusions = [], []
        with rdBase.BlockLogs():
            for index, graph in enumerate(records):
                try:
                    raw_molecule(graph)
                except (ValueError, TypeError, RuntimeError, OverflowError) as error:
                    exclusions.append({"index": index, "reason": f"{type(error).__name__}: {error}"})
                else:
                    valid_indices.append(index)
        print(f"Raw RDKit-valid: {len(valid_indices):,}; excluded: {len(exclusions):,}; requested: {count}", flush=True)
        if len(valid_indices) < count:
            hint = f" First exclusion: {exclusions[0]['reason']}" if exclusions else ""
            raise ValueError(f"Only {len(valid_indices)} drawable molecules for {count} panels. Reduce --row/--column or provide attributed molecular graphs.{hint}")
        indices = random.Random(args.seed).sample(valid_indices, count)
        canvas = Image.new("RGB", (args.cols * args.panel_width, args.rows * args.panel_height), "white")
        selected = []
        for panel, index in enumerate(indices):
            mol, smiles = raw_molecule(records[index])
            tile = draw_panel(mol, args, index)
            row, col = divmod(panel, args.cols)
            canvas.paste(tile, (col * args.panel_width, row * args.panel_height))
            selected.append({"index": index, "smiles": smiles, "row": row + 1, "column": col + 1})
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, format="PNG", dpi=(args.dpi, args.dpi))
        sidecar = output.with_name(output.stem + "_selection.json")
        sidecar.write_text(json.dumps({
            "dataset": args.dataset, "input_file": str(path), "output_png": str(output),
            "rows": args.rows, "columns": args.cols, "seed": args.seed,
            "total_records": len(records), "valid_records": len(valid_indices),
            "excluded_records": len(exclusions), "exclusions": exclusions, "selected": selected,
            "sampling": "uniform without replacement among raw RDKit-valid file indices; no deduplication",
            "charge_inference": False, "bond_correction": False, "connectedness_filter": False,
            "show_hydrogens": args.show_hydrogens, "show_indices": args.show_indices,
            "panel_width_pixels": args.panel_width, "panel_height_pixels": args.panel_height,
            "dpi": args.dpi, "rdkit_version": rdBase.rdkitVersion, "networkx_version": nx.__version__,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {output}\nSaved {sidecar}")
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, pickle.UnpicklingError, EOFError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
