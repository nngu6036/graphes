from __future__ import annotations

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder
from grapher.datasets.zinc_utils import zinc_preparation_error


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


class QM9DatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        try:
            from torch_geometric.datasets import QM9
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("QM9 preparation requires torch-geometric.") from exc
        dataset = QM9(root=str(self.config.get("pyg_root", "outputs/raw_datasets/qm9")))
        indices = list(range(len(dataset)))
        if self.config.get("shuffle", True):
            self.rng.shuffle(indices)
        max_graphs = self.config.get("max_graphs")
        if max_graphs is not None:
            indices = indices[: int(max_graphs)]
        graphs = [
            _pyg_data_to_graph(
                dataset[index],
                include_targets=bool(self.config.get("include_targets", True)),
            )
            for index in indices
        ]
        for graph in graphs:
            graph.graph["source_dataset"] = "qm9"
        return self.finalize(graphs, shuffle=False)


class ZINCDatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        raise zinc_preparation_error("build ZINC from PyG")
