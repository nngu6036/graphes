from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import networkx as nx
import numpy as np
import torch

from grapher.hybrid.data import (
    GraphCategoryVocabulary,
    GraphletBasis,
    HybridEndpointExample,
    collate_endpoint_examples,
    graph_to_categorical_arrays,
)
from grapher.hybrid.model import HybridEndpointPredictor
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.refinement.rewiring import (
    Action,
    enumerate_valid_double_edge_swaps,
    sample_valid_double_edge_swaps,
)


@dataclass
class HybridPrediction:
    node_probabilities: np.ndarray
    edge_probabilities: np.ndarray
    sampled_node_labels: np.ndarray
    sampled_edge_labels: np.ndarray
    graphlet_history: dict[str, dict[str, float]]
    graphlet_connected_mass: dict[str, float]
    graphlet_mean_history: dict[str, dict[str, float]]
    graphlet_mean_connected_mass: dict[str, float]
    sampled_graph: nx.Graph
    sampled_degree_match: bool
    sampled_connected: bool


@dataclass(frozen=True)
class HybridRefinerConfig:
    steps: int = 40
    candidate_budget: int = 64
    preserve_connectivity: bool = True
    selection: str = "greedy"
    temperature: float = 0.1
    categorical_weight: float = 1.0
    probability_weight: float = 0.0
    graphlet_weight: float = 1.0
    graphlet_mass_weight: float = 0.25
    graphlet_top_k: int = 8
    accept_only_improving: bool = True
    min_improvement: float = 1.0e-8
    patience: int = 2
    sample_endpoint: bool = True
    sample_graphlet: bool = True
    endpoint_temperature: float = 1.0
    refresh_prediction_every: int = 1
    infeasible_target_policy: str = "guidance_only"
    report_guidance_consistency: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None = None,
    ) -> "HybridRefinerConfig":
        data = data or {}
        config = cls(
            steps=int(data.get("steps", 40)),
            candidate_budget=int(data.get("candidate_budget", 64)),
            preserve_connectivity=bool(data.get("preserve_connectivity", True)),
            selection=str(data.get("selection", "greedy")).lower(),
            temperature=float(data.get("temperature", 0.1)),
            categorical_weight=float(data.get("categorical_weight", 1.0)),
            probability_weight=float(data.get("probability_weight", 0.0)),
            graphlet_weight=float(data.get("graphlet_weight", 1.0)),
            graphlet_mass_weight=float(data.get("graphlet_mass_weight", 0.25)),
            graphlet_top_k=int(data.get("graphlet_top_k", 8)),
            accept_only_improving=bool(
                data.get("accept_only_improving", True)
            ),
            min_improvement=float(data.get("min_improvement", 1.0e-8)),
            patience=max(int(data.get("patience", 2)), 1),
            sample_endpoint=bool(data.get("sample_endpoint", True)),
            sample_graphlet=bool(data.get("sample_graphlet", True)),
            endpoint_temperature=float(data.get("endpoint_temperature", 1.0)),
            refresh_prediction_every=max(
                int(data.get("refresh_prediction_every", 1)),
                1,
            ),
            infeasible_target_policy=str(
                data.get("infeasible_target_policy", "guidance_only")
            ).lower(),
            report_guidance_consistency=bool(
                data.get("report_guidance_consistency", False)
            ),
        )
        if config.steps < 0:
            raise ValueError("hybrid_refiner.steps must be non-negative.")
        if config.candidate_budget == 0:
            raise ValueError(
                "hybrid_refiner.candidate_budget must be positive, or negative "
                "to request complete enumeration."
            )
        if config.temperature <= 0.0 or config.endpoint_temperature <= 0.0:
            raise ValueError("Hybrid temperatures must be positive.")
        if config.selection not in {"greedy", "argmax", "softmax", "sample"}:
            raise ValueError(f"Unknown hybrid selection {config.selection!r}.")
        if config.infeasible_target_policy != "guidance_only":
            raise ValueError(
                "Only infeasible_target_policy='guidance_only' is implemented. "
                "The sampled endpoint is never installed directly."
            )
        if (
            config.categorical_weight == 0.0
            and config.probability_weight == 0.0
            and config.graphlet_weight == 0.0
        ):
            raise ValueError("At least one hybrid guidance weight must be non-zero.")
        return config


def _current_only_batch(
    graph: nx.Graph,
    *,
    time: float,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
):
    example = HybridEndpointExample(
        current_graph=graph,
        target_graph=graph,
        time=float(time),
        graphlet_target=np.zeros(graphlet_basis.width, dtype=np.float32),
        graphlet_mass_target=np.zeros(
            len(graphlet_basis.sizes),
            dtype=np.float32,
        ),
    )
    return collate_endpoint_examples([example], vocabulary)


def _sample_categorical_rows(
    probabilities: np.ndarray,
    rng: np.random.Generator,
    *,
    sample: bool,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not sample:
        return np.argmax(probabilities, axis=-1).astype(np.int64)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    values = np.zeros(flat.shape[0], dtype=np.int64)
    for index, row in enumerate(flat):
        row = np.maximum(row, 0.0)
        total = float(row.sum())
        if total <= 0.0:
            row = np.full(row.size, 1.0 / max(row.size, 1))
        else:
            row = row / total
        values[index] = int(rng.choice(row.size, p=row))
    return values.reshape(probabilities.shape[:-1])


def _sample_endpoint_labels(
    node_probabilities: np.ndarray,
    edge_probabilities: np.ndarray,
    rng: np.random.Generator,
    *,
    sample: bool,
) -> tuple[np.ndarray, np.ndarray]:
    node_labels = _sample_categorical_rows(
        node_probabilities,
        rng,
        sample=sample,
    )
    node_count = node_labels.shape[0]
    edge_labels = np.zeros((node_count, node_count), dtype=np.int64)
    for u in range(node_count):
        for v in range(u + 1, node_count):
            probabilities = edge_probabilities[u, v]
            if sample:
                probabilities = np.maximum(probabilities, 0.0)
                probabilities = probabilities / max(
                    float(probabilities.sum()),
                    1.0e-12,
                )
                category = int(rng.choice(probabilities.size, p=probabilities))
            else:
                category = int(np.argmax(probabilities))
            edge_labels[u, v] = category
            edge_labels[v, u] = category
    return node_labels, edge_labels


def graph_from_categorical_labels(
    node_labels: np.ndarray,
    edge_labels: np.ndarray,
    vocabulary: GraphCategoryVocabulary,
) -> nx.Graph:
    node_labels = np.asarray(node_labels, dtype=np.int64).reshape(-1)
    edge_labels = np.asarray(edge_labels, dtype=np.int64)
    graph = nx.Graph()
    for node, category in enumerate(node_labels):
        attributes = {}
        if vocabulary.node_attribute:
            attributes[vocabulary.node_attribute] = vocabulary.node_value(
                int(category)
            )
        graph.add_node(int(node), **attributes)
    for u in range(node_labels.size):
        for v in range(u + 1, node_labels.size):
            category = int(edge_labels[u, v])
            if category <= 0:
                continue
            attributes = {}
            if vocabulary.edge_attribute:
                attributes[vocabulary.edge_attribute] = vocabulary.edge_value(
                    category
                )
            graph.add_edge(u, v, **attributes)
    return graph


@torch.no_grad()
def predict_hybrid_target(
    model: HybridEndpointPredictor,
    graph: nx.Graph,
    *,
    time: float,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    device: torch.device | str,
    rng: np.random.Generator,
    sample_endpoint: bool = True,
    sample_graphlet: bool = True,
    endpoint_temperature: float = 1.0,
) -> HybridPrediction:
    model.eval()
    batch = _current_only_batch(
        graph,
        time=time,
        vocabulary=vocabulary,
        graphlet_basis=graphlet_basis,
    ).to(device)
    outputs = model(batch)
    node_probabilities_t, edge_probabilities_t = model.endpoint_probabilities(
        outputs,
        temperature=endpoint_temperature,
    )
    graphlet_mean_t, graphlet_mass_mean_t = model.graphlet_means(outputs)
    node_probabilities = node_probabilities_t[0].detach().cpu().numpy()
    edge_probabilities = edge_probabilities_t[0].detach().cpu().numpy()
    node_labels, edge_labels = _sample_endpoint_labels(
        node_probabilities,
        edge_probabilities,
        rng,
        sample=sample_endpoint,
    )

    alpha = outputs["graphlet_alpha"][0].detach().cpu().numpy()
    graphlet_sample = np.zeros(graphlet_basis.width, dtype=np.float64)
    for start, stop in graphlet_basis.slices:
        block = np.maximum(alpha[start:stop], 1.0e-12)
        graphlet_sample[start:stop] = (
            rng.dirichlet(block)
            if sample_graphlet
            else block / float(block.sum())
        )
    mass_ab = outputs["graphlet_mass_ab"][0].detach().cpu().numpy()
    graphlet_mass = np.asarray(
        [
            rng.beta(max(float(a), 1.0e-12), max(float(b), 1.0e-12))
            if sample_graphlet
            else float(a / max(a + b, 1.0e-12))
            for a, b in mass_ab
        ],
        dtype=np.float64,
    )
    sampled_graph = graph_from_categorical_labels(
        node_labels,
        edge_labels,
        vocabulary,
    )
    current_degree = [int(graph.degree(node)) for node in sorted(graph.nodes())]
    sampled_degree = [
        int(sampled_graph.degree(node)) for node in sorted(sampled_graph.nodes())
    ]
    return HybridPrediction(
        node_probabilities=node_probabilities,
        edge_probabilities=edge_probabilities,
        sampled_node_labels=node_labels,
        sampled_edge_labels=edge_labels,
        graphlet_history=graphlet_basis.unflatten_history(graphlet_sample),
        graphlet_connected_mass={
            k: float(value)
            for k, value in zip(graphlet_basis.sizes, graphlet_mass)
        },
        graphlet_mean_history=graphlet_basis.unflatten_history(
            graphlet_mean_t[0].detach().cpu().numpy()
        ),
        graphlet_mean_connected_mass={
            k: float(value)
            for k, value in zip(
                graphlet_basis.sizes,
                graphlet_mass_mean_t[0].detach().cpu().numpy(),
            )
        },
        sampled_graph=sampled_graph,
        sampled_degree_match=current_degree == sampled_degree,
        sampled_connected=(
            nx.is_connected(sampled_graph)
            if sampled_graph.number_of_nodes() > 0
            else False
        ),
    )


def _edge_after_category(
    u: int,
    v: int,
    prediction: HybridPrediction,
) -> int:
    sampled = int(prediction.sampled_edge_labels[u, v])
    if sampled > 0:
        return sampled
    probabilities = prediction.edge_probabilities[u, v].copy()
    probabilities[0] = -1.0
    return int(np.argmax(probabilities))


def apply_hybrid_action(
    graph: nx.Graph,
    action: Action,
    prediction: HybridPrediction,
    vocabulary: GraphCategoryVocabulary,
) -> nx.Graph:
    removed, added = action
    candidate = graph.copy()
    for u, v in removed:
        candidate.remove_edge(u, v)
    for u, v in added:
        category = _edge_after_category(u, v, prediction)
        attributes = {}
        if vocabulary.edge_attribute:
            attributes[vocabulary.edge_attribute] = vocabulary.edge_value(
                category
            )
        candidate.add_edge(u, v, **attributes)
    return candidate


def apply_hybrid_attributes(
    graph: nx.Graph,
    prediction: HybridPrediction,
    vocabulary: GraphCategoryVocabulary,
) -> nx.Graph:
    """Materialize predicted categories on a fixed generated topology.

    The sampled endpoint is only guidance because its topology can violate the
    requested degree sequence.  Its node and present-edge categories, however,
    are the generated molecular attributes and must be installed on the
    degree-preserving topology returned by the refiner.
    """

    attributed = graph.copy()
    if vocabulary.node_attribute:
        for node in attributed.nodes():
            category = int(prediction.sampled_node_labels[int(node)])
            attributed.nodes[node][vocabulary.node_attribute] = (
                vocabulary.node_value(category)
            )
    if vocabulary.edge_attribute:
        for u, v in attributed.edges():
            category = _edge_after_category(
                int(u),
                int(v),
                prediction,
            )
            attributed.edges[u, v][vocabulary.edge_attribute] = (
                vocabulary.edge_value(category)
            )
    return attributed


def _categorical_gain(
    graph: nx.Graph,
    action: Action,
    prediction: HybridPrediction,
    vocabulary: GraphCategoryVocabulary,
) -> tuple[float, float]:
    _, current_edges = graph_to_categorical_arrays(graph, vocabulary)
    removed, added = action
    changed = list(removed) + list(added)
    mismatch_before = 0.0
    mismatch_after = 0.0
    probability_gain = 0.0
    for u, v in changed:
        before = int(current_edges[u, v])
        after = (
            0
            if (u, v) in removed
            else _edge_after_category(u, v, prediction)
        )
        target = int(prediction.sampled_edge_labels[u, v])
        mismatch_before += float(before != target)
        mismatch_after += float(after != target)
        probabilities = prediction.edge_probabilities[u, v]
        probability_gain += float(
            np.log(max(float(probabilities[after]), 1.0e-12))
            - np.log(max(float(probabilities[before]), 1.0e-12))
        )
    denominator = max(len(changed), 1)
    return (
        float((mismatch_before - mismatch_after) / denominator),
        float(probability_gain / denominator),
    )


def _graphlet_distance_from_summary(
    summary: dict[str, Any],
    prediction: HybridPrediction,
    graphlet_basis: GraphletBasis,
    *,
    mass_weight: float,
) -> float:
    current = graphlet_basis.flatten_history(
        summary.get("graphlet_history", {}) or {}
    ).astype(np.float64)
    target = graphlet_basis.flatten_history(
        prediction.graphlet_history
    ).astype(np.float64)
    block_distances = []
    for start, stop in graphlet_basis.slices:
        block_distances.append(
            float(np.linalg.norm(current[start:stop] - target[start:stop]))
        )
    current_mass = graphlet_basis.flatten_mass(
        summary.get("graphlet_connected_mass", {}) or {}
    ).astype(np.float64)
    target_mass = graphlet_basis.flatten_mass(
        prediction.graphlet_connected_mass
    ).astype(np.float64)
    graphlet_distance = (
        float(np.mean(block_distances)) if block_distances else 0.0
    )
    mass_distance = (
        float(np.mean(np.abs(current_mass - target_mass)))
        if current_mass.size
        else 0.0
    )
    return graphlet_distance + float(mass_weight) * mass_distance


def graphlet_guidance_distance(
    graph: nx.Graph,
    prediction: HybridPrediction,
    *,
    summary_config: SummaryConfig,
    graphlet_basis: GraphletBasis,
    mass_weight: float,
) -> float:
    return _graphlet_distance_from_summary(
        extract_summary(graph, summary_config),
        prediction,
        graphlet_basis,
        mass_weight=mass_weight,
    )


def score_hybrid_candidates(
    graph: nx.Graph,
    candidates: Sequence[Action],
    prediction: HybridPrediction,
    *,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    summary_config: SummaryConfig,
    config: HybridRefinerConfig | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = (
        config
        if isinstance(config, HybridRefinerConfig)
        else HybridRefinerConfig.from_dict(config)
    )
    rows: list[dict[str, Any]] = []
    for action in candidates:
        categorical_gain, probability_gain = _categorical_gain(
            graph,
            action,
            prediction,
            vocabulary,
        )
        rows.append(
            {
                "action": action,
                "categorical_gain": categorical_gain,
                "probability_gain": probability_gain,
                "node_guidance_gain": 0.0,
                "graphlet_gain": 0.0,
                "hybrid_score": (
                    cfg.categorical_weight * categorical_gain
                    + cfg.probability_weight * probability_gain
                ),
                "candidate_graph": None,
            }
        )

    if cfg.graphlet_weight != 0.0 and rows:
        has_pre_graphlet_signal = (
            cfg.categorical_weight != 0.0 or cfg.probability_weight != 0.0
        )
        if (
            not has_pre_graphlet_signal
            or cfg.graphlet_top_k <= 0
            or cfg.graphlet_top_k >= len(rows)
        ):
            graphlet_indices = list(range(len(rows)))
        else:
            preliminary = np.asarray(
                [float(row["hybrid_score"]) for row in rows],
                dtype=np.float64,
            )
            graphlet_indices = np.argsort(-preliminary)[
                : int(cfg.graphlet_top_k)
            ].tolist()
        current_distance = graphlet_guidance_distance(
            graph,
            prediction,
            summary_config=summary_config,
            graphlet_basis=graphlet_basis,
            mass_weight=cfg.graphlet_mass_weight,
        )
        shortlisted = set(int(index) for index in graphlet_indices)
        for index, row in enumerate(rows):
            if index not in shortlisted:
                row["hybrid_score"] = float("-inf")
                continue
            candidate_graph = apply_hybrid_action(
                graph,
                row["action"],
                prediction,
                vocabulary,
            )
            candidate_distance = graphlet_guidance_distance(
                candidate_graph,
                prediction,
                summary_config=summary_config,
                graphlet_basis=graphlet_basis,
                mass_weight=cfg.graphlet_mass_weight,
            )
            row["graphlet_gain"] = float(current_distance - candidate_distance)
            row["hybrid_score"] = float(
                row["hybrid_score"]
                + cfg.graphlet_weight * row["graphlet_gain"]
            )
            row["candidate_graph"] = candidate_graph
    for row in rows:
        if row["candidate_graph"] is None and np.isfinite(row["hybrid_score"]):
            row["candidate_graph"] = apply_hybrid_action(
                graph,
                row["action"],
                prediction,
                vocabulary,
            )
    return rows


def _select_scored_candidate(
    rows: list[dict[str, Any]],
    *,
    config: HybridRefinerConfig,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    finite = [row for row in rows if np.isfinite(float(row["hybrid_score"]))]
    if not finite:
        return None
    scores = np.asarray(
        [float(row["hybrid_score"]) for row in finite],
        dtype=np.float64,
    )
    if config.selection in {"greedy", "argmax"}:
        return finite[int(np.argmax(scores))]
    logits = scores / max(float(config.temperature), 1.0e-12)
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(probabilities.sum())
    return finite[int(rng.choice(len(finite), p=probabilities))]


def _nodewise_degrees(graph: nx.Graph) -> list[int]:
    return [int(graph.degree(node)) for node in sorted(graph.nodes())]


def refine_graph_with_hybrid_predictions(
    graph: nx.Graph,
    *,
    model: HybridEndpointPredictor,
    vocabulary: GraphCategoryVocabulary,
    graphlet_basis: GraphletBasis,
    summary_config: SummaryConfig,
    refiner_config: HybridRefinerConfig | dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
    rng: np.random.Generator | None = None,
    return_trace: bool = False,
    prediction_fn: Callable[..., HybridPrediction] | None = None,
) -> nx.Graph | tuple[nx.Graph, list[dict[str, Any]]]:
    cfg = (
        refiner_config
        if isinstance(refiner_config, HybridRefinerConfig)
        else HybridRefinerConfig.from_dict(refiner_config)
    )
    generator = rng if rng is not None else np.random.default_rng(0)
    predictor = prediction_fn or predict_hybrid_target
    current = nx.convert_node_labels_to_integers(
        nx.Graph(graph),
        ordering="sorted",
    )
    initial_degrees = _nodewise_degrees(current)
    trace: list[dict[str, Any]] = []
    prediction: HybridPrediction | None = None
    stalled = 0

    for step in range(cfg.steps):
        if prediction is None or step % cfg.refresh_prediction_every == 0:
            time = float(step / max(cfg.steps - 1, 1))
            prediction = predictor(
                model,
                current,
                time=time,
                vocabulary=vocabulary,
                graphlet_basis=graphlet_basis,
                device=device,
                rng=generator,
                sample_endpoint=cfg.sample_endpoint,
                sample_graphlet=cfg.sample_graphlet,
                endpoint_temperature=cfg.endpoint_temperature,
            )
        if cfg.candidate_budget < 0:
            candidates = enumerate_valid_double_edge_swaps(
                current,
                preserve_connectivity=cfg.preserve_connectivity,
            )
        else:
            candidates = sample_valid_double_edge_swaps(
                current,
                cfg.candidate_budget,
                generator,
                preserve_connectivity=cfg.preserve_connectivity,
            )
        if not candidates:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "no_candidates",
                    "sampled_target_degree_match": prediction.sampled_degree_match,
                }
            )
            break

        rows = score_hybrid_candidates(
            current,
            candidates,
            prediction,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            summary_config=summary_config,
            config=cfg,
        )
        chosen = _select_scored_candidate(rows, config=cfg, rng=generator)
        if chosen is None:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "no_scored_candidate",
                    "sampled_target_degree_match": prediction.sampled_degree_match,
                }
            )
            break
        score = float(chosen["hybrid_score"])
        if cfg.accept_only_improving and score <= cfg.min_improvement:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "no_improving_candidate",
                    "best_score": score,
                    "categorical_gain": float(chosen["categorical_gain"]),
                    "probability_gain": float(chosen["probability_gain"]),
                    "node_guidance_gain": 0.0,
                    "graphlet_gain": float(chosen["graphlet_gain"]),
                    "hybrid_score": score,
                    "sampled_target_degree_match": prediction.sampled_degree_match,
                }
            )
            break

        candidate_graph = chosen["candidate_graph"]
        if not isinstance(candidate_graph, nx.Graph):
            raise RuntimeError("Hybrid scorer did not materialize its candidate.")
        if _nodewise_degrees(candidate_graph) != initial_degrees:
            raise AssertionError("A hybrid rewiring action changed node degrees.")
        if (
            cfg.preserve_connectivity
            and candidate_graph.number_of_nodes() > 1
            and not nx.is_connected(candidate_graph)
        ):
            raise AssertionError("A hybrid rewiring action broke connectivity.")
        current = candidate_graph
        stalled = stalled + 1 if score <= cfg.min_improvement else 0

        consistency = None
        if cfg.report_guidance_consistency:
            consistency = graphlet_guidance_distance(
                prediction.sampled_graph,
                prediction,
                summary_config=summary_config,
                graphlet_basis=graphlet_basis,
                mass_weight=cfg.graphlet_mass_weight,
            )
        trace.append(
            {
                "step": step,
                "accepted": True,
                "num_candidates": len(candidates),
                "categorical_gain": float(chosen["categorical_gain"]),
                "probability_gain": float(chosen["probability_gain"]),
                "node_guidance_gain": 0.0,
                "graphlet_gain": float(chosen["graphlet_gain"]),
                "hybrid_score": score,
                "sampled_target_degree_match": prediction.sampled_degree_match,
                "sampled_target_connected": prediction.sampled_connected,
                "guidance_graphlet_disagreement": consistency,
            }
        )
        if stalled >= cfg.patience:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "patience_exhausted",
                }
            )
            break

    if vocabulary.node_attribute or vocabulary.edge_attribute:
        # Refresh once at the realized endpoint, then attach categories to
        # every node and edge.  Previously only edges introduced by accepted
        # swaps received labels, leaving QM9 outputs unusable by RDKit.
        prediction = predictor(
            model,
            current,
            time=1.0,
            vocabulary=vocabulary,
            graphlet_basis=graphlet_basis,
            device=device,
            rng=generator,
            sample_endpoint=cfg.sample_endpoint,
            sample_graphlet=cfg.sample_graphlet,
            endpoint_temperature=cfg.endpoint_temperature,
        )
        current = apply_hybrid_attributes(current, prediction, vocabulary)

    if return_trace:
        return current, trace
    return current
