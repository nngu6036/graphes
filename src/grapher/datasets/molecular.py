from __future__ import annotations

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder
from grapher.datasets.zinc_utils import zinc_preparation_error


def _pyg_data_to_graph(data, *, include_targets: bool = True) -> nx.Graph:
    graph = nx.Graph()
    x = getattr(data, "x", None)
    num_nodes = int(getattr(data, "num_nodes", 0) or (x.size(0) if x is not None else 0))
    for i in range(num_nodes):
        attrs = {}
        if x is not None:
            vals = x[i].detach().cpu().view(-1).tolist()
            attrs["feats"] = [float(v) for v in vals]
            if vals:
                attrs["node_label"] = int(vals[0])
                attrs["atomic_number"] = int(vals[0])
                attrs["z"] = int(vals[0])
        graph.add_node(i, **attrs)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if edge_index is not None:
        edges = edge_index.detach().cpu().numpy().T
        for j, (u, v) in enumerate(edges):
            if int(u) == int(v):
                continue
            attrs = {}
            if edge_attr is not None:
                vals = edge_attr[j].detach().cpu().view(-1).tolist()
                attrs["edge_attr"] = [float(v) for v in vals]
                if vals:
                    attrs["edge_type"] = int(vals[0])
            graph.add_edge(int(u), int(v), **attrs)
    if include_targets and hasattr(data, "y") and data.y is not None:
        graph.graph["graph_label"] = data.y.detach().cpu().view(-1).tolist()
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
            rng = self.rng
            rng.shuffle(indices)
        max_graphs = self.config.get("max_graphs")
        if max_graphs is not None:
            indices = indices[: int(max_graphs)]
        graphs = [_pyg_data_to_graph(dataset[i], include_targets=bool(self.config.get("include_targets", True))) for i in indices]
        for graph in graphs:
            graph.graph["source_dataset"] = "qm9"
        return self.finalize(graphs, shuffle=False)


class ZINCDatasetBuilder(BaseDatasetBuilder):
    def build(self) -> dict[str, list[nx.Graph]]:
        raise zinc_preparation_error("build ZINC from PyG")
