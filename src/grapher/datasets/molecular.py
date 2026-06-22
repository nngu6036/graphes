from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder
from grapher.datasets.zinc_utils import zinc_preparation_error
from grapher.utils.logging import get_logger


logger = get_logger(__name__)


def _qm9_bond_type(edge_attr_row) -> int:
    """Map PyG QM9 bond features to 1=single,2=double,3=triple,4=aromatic."""

    values = edge_attr_row.detach().cpu().reshape(-1).tolist()
    if not values:
        return 1
    if len(values) == 1:
        value = int(round(float(values[0])))
        return value if value >= 1 else 1
    # PyG QM9 stores a one-hot bond vector in the conventional order
    # single/double/triple/aromatic.
    return int(max(range(len(values)), key=lambda index: float(values[index]))) + 1


def _pyg_data_to_graph(data, *, include_targets: bool = True) -> nx.Graph:
    graph = nx.Graph()
    x = getattr(data, "x", None)
    z = getattr(data, "z", None)
    num_nodes = int(
        getattr(data, "num_nodes", 0)
        or (z.size(0) if z is not None else 0)
        or (x.size(0) if x is not None else 0)
    )
    for index in range(num_nodes):
        attrs = {}
        if x is not None:
            values = x[index].detach().cpu().reshape(-1).tolist()
            attrs["feats"] = [float(value) for value in values]
        atomic_number = None
        if z is not None:
            atomic_number = int(z[index].detach().cpu().item())
        elif x is not None:
            values = x[index].detach().cpu().reshape(-1).tolist()
            if values:
                atomic_number = int(round(float(values[0])))
        if atomic_number is not None:
            attrs["node_label"] = f"atomic_number={atomic_number}"
            attrs["atomic_number"] = int(atomic_number)
            attrs["z"] = int(atomic_number)
        graph.add_node(index, **attrs)

    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if edge_index is not None:
        edges = edge_index.detach().cpu().numpy().T
        for edge_index_row, (u, v) in enumerate(edges):
            u, v = int(u), int(v)
            if u == v:
                continue
            edge_type = 1
            raw_features: list[float] = [1.0]
            if edge_attr is not None:
                raw_features = [
                    float(value)
                    for value in edge_attr[edge_index_row]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .tolist()
                ]
                edge_type = _qm9_bond_type(edge_attr[edge_index_row])
            graph.add_edge(
                u,
                v,
                edge_type=int(edge_type),
                edge_attr=raw_features,
                bond_order=1.5 if int(edge_type) == 4 else float(edge_type),
            )
    if include_targets and hasattr(data, "y") and data.y is not None:
        graph.graph["graph_label"] = data.y.detach().cpu().reshape(-1).tolist()
    return graph


def _bond_type_id(bond: Any) -> int | None:
    from rdkit import Chem

    mapping = {
        Chem.BondType.SINGLE: 1,
        Chem.BondType.DOUBLE: 2,
        Chem.BondType.TRIPLE: 3,
        Chem.BondType.AROMATIC: 4,
    }
    return mapping.get(bond.GetBondType())


def _find_qm9_sdf(root: str | Path) -> Path | None:
    raw_dir = Path(root) / "raw"
    candidates = [
        raw_dir / "gdb9.sdf",
        raw_dir / "qm9.sdf",
    ]
    candidates.extend(sorted(raw_dir.glob("*.sdf")))
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_qm9_targets(root: str | Path) -> list[list[float]] | None:
    raw_dir = Path(root) / "raw"
    candidates = [
        raw_dir / "gdb9.sdf.csv",
        raw_dir / "qm9.csv",
    ]
    candidates.extend(sorted(raw_dir.glob("*.csv")))
    for path in candidates:
        if not path.exists():
            continue
        rows: list[list[float]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            first = next(reader, None)
            if first is None:
                continue
            data_rows = reader if any(_is_float(cell) for cell in first) else reader
            if any(_is_float(cell) for cell in first):
                rows.append(_numeric_csv_row(first))
            for row in data_rows:
                values = _numeric_csv_row(row)
                if values:
                    rows.append(values)
        if rows:
            return rows
    return None


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _numeric_csv_row(row: list[str]) -> list[float]:
    values = []
    for cell in row:
        try:
            values.append(float(cell))
        except (TypeError, ValueError):
            continue
    return values


def _rdkit_mol_to_graph(mol: Any, *, target: list[float] | None, source_index: int, include_positions: bool) -> nx.Graph | None:
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        z = int(atom.GetAtomicNum())
        if z <= 0:
            return None
        attrs: dict[str, Any] = {
            "node_label": f"atomic_number={z}",
            "atomic_number": z,
            "z": z,
            "atom_symbol": str(atom.GetSymbol()),
            "feats": [float(z)],
        }
        graph.add_node(idx, **attrs)

    for bond in mol.GetBonds():
        edge_type = _bond_type_id(bond)
        if edge_type is None:
            return None
        graph.add_edge(
            int(bond.GetBeginAtomIdx()),
            int(bond.GetEndAtomIdx()),
            edge_type=int(edge_type),
            edge_attr=[float(edge_type)],
            bond_order=1.5 if int(edge_type) == 4 else float(edge_type),
            bond_type_name=str(bond.GetBondType()),
        )

    if include_positions and mol.GetNumConformers() > 0:
        conformer = mol.GetConformer()
        for node in graph.nodes:
            pos = conformer.GetAtomPosition(int(node))
            graph.nodes[node]["pos"] = [float(pos.x), float(pos.y), float(pos.z)]

    graph.graph["source_dataset"] = "qm9"
    graph.graph["source_index"] = int(source_index)
    if target is not None:
        graph.graph["graph_label"] = [float(value) for value in target]
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def _rdkit_qm9_graphs(root: str | Path, *, include_targets: bool, include_positions: bool) -> list[nx.Graph] | None:
    sdf_path = _find_qm9_sdf(root)
    if sdf_path is None:
        return None
    try:
        from rdkit import Chem, RDLogger
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("QM9 SDF fallback requires RDKit.") from exc

    RDLogger.DisableLog("rdApp.*")
    targets = _load_qm9_targets(root) if include_targets else None
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
    graphs: list[nx.Graph] = []
    skipped = 0
    for index, mol in enumerate(supplier):
        if mol is None:
            skipped += 1
            continue
        target = targets[index] if targets is not None and index < len(targets) else None
        graph = _rdkit_mol_to_graph(mol, target=target, source_index=index, include_positions=include_positions)
        if graph is None:
            skipped += 1
            continue
        graphs.append(graph)
    logger.info("Loaded QM9 from SDF %s graphs=%d skipped=%d", sdf_path, len(graphs), skipped)
    return graphs


class QM9DatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        pyg_root = str(self.config.get("pyg_root", "outputs/raw_datasets/qm9"))
        include_targets = bool(self.config.get("include_targets", True))
        include_positions = bool(self.config.get("include_positions", False))
        prefer_preprocessed = bool(self.config.get("prefer_preprocessed", True))

        if prefer_preprocessed:
            graphs = _rdkit_qm9_graphs(pyg_root, include_targets=include_targets, include_positions=include_positions)
            if graphs is not None:
                return self._finalize_graphs(graphs)

        try:
            from torch_geometric.datasets import QM9
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("QM9 preparation requires torch-geometric.") from exc

        try:
            dataset = QM9(root=pyg_root)
        except Exception as exc:
            graphs = _rdkit_qm9_graphs(pyg_root, include_targets=include_targets, include_positions=include_positions)
            if graphs is None:
                raise
            logger.warning("PyG QM9 processing failed; using RDKit SDF fallback instead. error=%s", exc)
            return self._finalize_graphs(graphs)

        indices = list(range(len(dataset)))
        if self.config.get("shuffle", True):
            self.rng.shuffle(indices)
        max_graphs = self.config.get("max_graphs")
        if max_graphs is not None:
            indices = indices[: int(max_graphs)]
        graphs = [
            _pyg_data_to_graph(
                dataset[index],
                include_targets=include_targets,
            )
            for index in indices
        ]
        for graph in graphs:
            graph.graph["source_dataset"] = "qm9"
        return self.finalize(graphs, shuffle=False)

    def _finalize_graphs(self, graphs: list[nx.Graph]) -> dict[str, list[nx.Graph]]:
        indices = list(range(len(graphs)))
        if self.config.get("shuffle", True):
            self.rng.shuffle(indices)
        max_graphs = self.config.get("max_graphs")
        if max_graphs is not None:
            indices = indices[: int(max_graphs)]
        selected = [graphs[int(index)] for index in indices]
        return self.finalize(selected, shuffle=False)


class ZINCDatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        raise zinc_preparation_error("build ZINC from PyG")
