from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder


class EgoCiteseerDatasetBuilder(BaseDatasetBuilder):
    def _candidate_raw_paths(self) -> list[Path]:
        out = []
        raw_path = self.config.get("raw_graph_path")
        if raw_path:
            out.append(Path(raw_path))
        root = Path(self.config.get("pyg_root", "outputs/raw_datasets/citeseer"))
        out.extend(
            [
                root / "ind.citeseer.graph",
                root / "CiteSeer" / "raw" / "ind.citeseer.graph",
                Path("outputs/raw_datasets/citeseer/CiteSeer/raw/ind.citeseer.graph"),
            ]
        )
        return out

    def _load_pickle_graph(self) -> nx.Graph | None:
        for path in self._candidate_raw_paths():
            if not path.exists():
                continue
            with open(path, "rb") as f:
                payload: Any = pickle.load(f, encoding="latin1")
            if isinstance(payload, nx.Graph):
                return nx.Graph(payload)
            if isinstance(payload, dict):
                return nx.Graph(payload)
        return None

    def _load_pyg_graph(self) -> nx.Graph:
        try:
            from torch_geometric.datasets import Planetoid
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ego_citeseer requires torch-geometric when no local ind.citeseer.graph pickle is available."
            ) from exc
        dataset = Planetoid(root=str(self.config.get("pyg_root", "outputs/raw_datasets/citeseer")), name="CiteSeer")
        data = dataset[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(int(data.num_nodes)))
        edge_index = data.edge_index.cpu().numpy()
        graph.add_edges_from((int(u), int(v)) for u, v in edge_index.T if int(u) != int(v))
        return graph

    def build(self) -> dict[str, list[nx.Graph]]:
        source = self._load_pickle_graph()
        if source is None:
            source = self._load_pyg_graph()
        if self.config.get("largest_connected_component", True) and source.number_of_nodes():
            nodes = max(nx.connected_components(source), key=len)
            source = source.subgraph(nodes).copy()

        radius = int(self.config.get("radius", 3))
        min_nodes = int(self.config.get("min_nodes", 4))
        max_nodes = int(self.config.get("max_nodes", 18))
        candidates = []
        for node in source.nodes:
            nodes = nx.single_source_shortest_path_length(source, node, cutoff=radius).keys()
            ego = source.subgraph(nodes).copy()
            if min_nodes <= ego.number_of_nodes() <= max_nodes:
                ego.graph["source_dataset"] = "ego_citeseer"
                candidates.append(ego)

        rng = self.rng
        rng.shuffle(candidates)
        requested = int(self.config.get("num_graphs", len(candidates)))
        if requested > len(candidates) and self.config.get("strict_num_graphs", False):
            raise ValueError(f"Requested {requested} ego_citeseer graphs but only {len(candidates)} candidates are available.")
        return self.finalize(candidates[: min(requested, len(candidates))], shuffle=bool(self.config.get("shuffle", True)))
