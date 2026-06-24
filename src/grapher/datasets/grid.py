from __future__ import annotations

import networkx as nx

from grapher.datasets.base import BaseDatasetBuilder


def _int_or_none(value) -> int | None:
    if value in (None, "", "none", "None"):
        return None
    return int(value)


class GridDatasetBuilder(BaseDatasetBuilder):
    """Generate rectangular 2D grid graphs."""

    def build(self) -> dict[str, list[nx.Graph]]:
        num_graphs = int(self.config.get("num_graphs", 1024))
        rows = _int_or_none(self.config.get("rows"))
        cols = _int_or_none(self.config.get("cols"))
        min_rows = int(self.config.get("min_rows", rows or 8))
        max_rows = int(self.config.get("max_rows", rows or 16))
        min_cols = int(self.config.get("min_cols", cols or min_rows))
        max_cols = int(self.config.get("max_cols", cols or max_rows))
        require_square = bool(self.config.get("require_square", False))

        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}.")
        if min_rows <= 0 or max_rows <= 0 or min_cols <= 0 or max_cols <= 0:
            raise ValueError("Grid row/column bounds must be positive.")
        if min_rows > max_rows:
            raise ValueError(f"min_rows={min_rows} cannot exceed max_rows={max_rows}.")
        if min_cols > max_cols:
            raise ValueError(f"min_cols={min_cols} cannot exceed max_cols={max_cols}.")

        rng = self.rng
        graphs: list[nx.Graph] = []
        for index in range(num_graphs):
            if rows is not None:
                row_count = rows
            else:
                row_count = int(rng.integers(min_rows, max_rows + 1))
            if cols is not None:
                col_count = cols
            elif require_square:
                col_count = row_count
            else:
                col_count = int(rng.integers(min_cols, max_cols + 1))

            graph = nx.grid_2d_graph(row_count, col_count)
            graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
            graph.graph["source_dataset"] = "grid"
            graph.graph["grid_index"] = int(index)
            graph.graph["grid_rows"] = int(row_count)
            graph.graph["grid_cols"] = int(col_count)
            graphs.append(graph)

        return self.finalize(graphs, shuffle=True)
