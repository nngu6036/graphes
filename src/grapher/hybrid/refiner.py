from __future__ import annotations

from collections import defaultdict
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
from grapher.molecular.constraints import (
    DEFAULT_GENERATED_BOND_TYPES,
    bond_order,
    is_molecular_valence_feasible,
)
from grapher.properties.summary import SummaryConfig, extract_summary
from grapher.refinement.rewiring import (
    Action,
    candidate_actions_from_edge_pair,
    enumerate_valid_double_edge_swaps,
    is_valid_action,
    sample_valid_double_edge_swaps,
)


@dataclass
class HybridPrediction:
    edge_probabilities: np.ndarray
    sampled_edge_labels: np.ndarray
    graphlet_history: dict[str, dict[str, float]]
    graphlet_connected_mass: dict[str, float]
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
    require_same_edge_type_pair: bool = False
    preserve_removed_edge_type: bool = False
    enforce_molecular_valence: bool = False
    molecular_allowed_bond_types: tuple[int, ...] = DEFAULT_GENERATED_BOND_TYPES
    molecular_candidate_attempt_multiplier: int = 2
    rdkit_candidate_check: bool = False

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
            accept_only_improving=bool(data.get("accept_only_improving", True)),
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
            require_same_edge_type_pair=bool(
                data.get("require_same_edge_type_pair", False)
            ),
            preserve_removed_edge_type=bool(
                data.get("preserve_removed_edge_type", False)
            ),
            enforce_molecular_valence=bool(
                data.get("enforce_molecular_valence", False)
            ),
            molecular_allowed_bond_types=tuple(
                int(value)
                for value in data.get(
                    "molecular_allowed_bond_types",
                    DEFAULT_GENERATED_BOND_TYPES,
                )
            ),
            molecular_candidate_attempt_multiplier=max(
                int(
                    data.get(
                        "molecular_candidate_attempt_multiplier",
                        data.get("molecular_candidate_oversample", 2),
                    )
                ),
                1,
            ),
            rdkit_candidate_check=bool(data.get("rdkit_candidate_check", False)),
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
        if config.require_same_edge_type_pair != config.preserve_removed_edge_type:
            raise ValueError(
                "Strict typed rewiring requires both require_same_edge_type_pair "
                "and preserve_removed_edge_type to have the same value."
            )
        if (
            config.categorical_weight == 0.0
            and config.probability_weight == 0.0
            and config.graphlet_weight == 0.0
        ):
            raise ValueError("At least one hybrid guidance weight must be non-zero.")
        return config


def _sample_endpoint_labels(
    node_probabilities: np.ndarray,
    edge_probabilities: np.ndarray,
    rng: np.random.Generator,
    *,
    sample: bool,
) -> tuple[np.ndarray, np.ndarray]:
    node_probabilities = np.asarray(node_probabilities, dtype=np.float64)
    if sample:
        node_labels = np.zeros(node_probabilities.shape[0], dtype=np.int64)
        for index, probabilities in enumerate(node_probabilities):
            probabilities = np.maximum(probabilities, 0.0)
            total = float(probabilities.sum())
            probabilities = (
                probabilities / total
                if total > 0.0
                else np.full(
                    probabilities.size,
                    1.0 / max(probabilities.size, 1),
                )
            )
            node_labels[index] = int(rng.choice(probabilities.size, p=probabilities))
    else:
        node_labels = np.argmax(node_probabilities, axis=-1).astype(np.int64)
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
            attributes[vocabulary.node_attribute] = vocabulary.node_value(int(category))
        graph.add_node(int(node), **attributes)
    for u in range(node_labels.size):
        for v in range(u + 1, node_labels.size):
            category = int(edge_labels[u, v])
            if category <= 0:
                continue
            attributes = {}
            if vocabulary.edge_attribute:
                attributes[vocabulary.edge_attribute] = vocabulary.edge_value(category)
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
    batch = collate_endpoint_examples(
        [
            HybridEndpointExample(
                current_graph=graph,
                target_graph=graph,
                time=float(time),
                graphlet_target=np.zeros(
                    graphlet_basis.width,
                    dtype=np.float32,
                ),
                graphlet_mass_target=np.zeros(
                    len(graphlet_basis.sizes),
                    dtype=np.float32,
                ),
            )
        ],
        vocabulary,
    ).to(device)
    outputs = model(batch)
    node_probabilities_t, edge_probabilities_t = model.endpoint_probabilities(
        outputs,
        temperature=endpoint_temperature,
    )
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
            rng.dirichlet(block) if sample_graphlet else block / float(block.sum())
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
        edge_probabilities=edge_probabilities,
        sampled_edge_labels=edge_labels,
        graphlet_history=graphlet_basis.unflatten_history(graphlet_sample),
        graphlet_connected_mass={
            k: float(value) for k, value in zip(graphlet_basis.sizes, graphlet_mass)
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


def _removed_edge_category(
    graph: nx.Graph,
    action: Action,
    vocabulary: GraphCategoryVocabulary,
) -> int | None:
    removed, _ = action
    categories = [int(vocabulary.edge_index(graph.edges[u, v])) for u, v in removed]
    if not categories or len(set(categories)) != 1:
        return None
    return categories[0]


def _sample_same_edge_type_actions(
    graph: nx.Graph,
    budget: int,
    rng: np.random.Generator,
    vocabulary: GraphCategoryVocabulary,
    *,
    preserve_connectivity: bool,
    attempt_multiplier: int,
) -> list[Action]:
    edge_groups: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        edge_groups[int(vocabulary.edge_index(data))].append(
            (min(int(u), int(v)), max(int(u), int(v)))
        )
    groups = [edges for edges in edge_groups.values() if len(edges) >= 2]
    if budget <= 0 or not groups:
        return []
    theoretical_max = sum(len(edges) * (len(edges) - 1) for edges in groups)
    target = min(int(budget), int(theoretical_max))
    group_weights = np.asarray(
        [len(edges) * (len(edges) - 1) / 2 for edges in groups],
        dtype=np.float64,
    )
    group_weights /= float(group_weights.sum())
    max_attempts = max(100, target * 25 * int(attempt_multiplier))
    out: list[Action] = []
    seen: set[Action] = set()
    for _ in range(max_attempts):
        edges = groups[int(rng.choice(len(groups), p=group_weights))]
        indices = rng.choice(len(edges), size=2, replace=False)
        actions = candidate_actions_from_edge_pair(
            edges[int(indices[0])],
            edges[int(indices[1])],
        )
        rng.shuffle(actions)
        for action in actions:
            if action in seen:
                continue
            if is_valid_action(
                graph,
                action,
                preserve_connectivity=preserve_connectivity,
            ):
                out.append(action)
                seen.add(action)
                if len(out) >= target:
                    return out
    return out


def apply_hybrid_action(
    graph: nx.Graph,
    action: Action,
    prediction: HybridPrediction,
    vocabulary: GraphCategoryVocabulary,
    *,
    preserve_removed_edge_type: bool = False,
) -> nx.Graph:
    removed, added = action
    preserved_category = (
        _removed_edge_category(graph, action, vocabulary)
        if preserve_removed_edge_type
        else None
    )
    if preserve_removed_edge_type and preserved_category is None:
        raise ValueError(
            "A typed molecular swap must remove two edges of the same type."
        )
    candidate = graph.copy()
    for u, v in removed:
        candidate.remove_edge(u, v)
    for u, v in added:
        category = (
            int(preserved_category)
            if preserved_category is not None
            else _edge_after_category(u, v, prediction)
        )
        attributes = {}
        if vocabulary.edge_attribute:
            value = vocabulary.edge_value(category)
            attributes[vocabulary.edge_attribute] = value
            if vocabulary.edge_attribute == "bond_type":
                attributes["bond_order"] = bond_order(int(value))
        candidate.add_edge(u, v, **attributes)
    return candidate


def _categorical_gain(
    graph: nx.Graph,
    action: Action,
    prediction: HybridPrediction,
    vocabulary: GraphCategoryVocabulary,
    *,
    preserve_removed_edge_type: bool = False,
) -> tuple[float, float]:
    _, current_edges = graph_to_categorical_arrays(graph, vocabulary)
    removed, added = action
    changed = list(removed) + list(added)
    mismatch_before = 0.0
    mismatch_after = 0.0
    probability_gain = 0.0
    preserved_category = (
        _removed_edge_category(graph, action, vocabulary)
        if preserve_removed_edge_type
        else None
    )
    for u, v in changed:
        before = int(current_edges[u, v])
        after = (
            0
            if (u, v) in removed
            else (
                int(preserved_category)
                if preserved_category is not None
                else _edge_after_category(u, v, prediction)
            )
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


def graphlet_guidance_distance(
    graph: nx.Graph,
    prediction: HybridPrediction,
    *,
    summary_config: SummaryConfig,
    graphlet_basis: GraphletBasis,
    mass_weight: float,
) -> float:
    summary = extract_summary(graph, summary_config)
    current = graphlet_basis.flatten_history(
        summary.get("graphlet_history", {}) or {}
    ).astype(np.float64)
    target = graphlet_basis.flatten_history(prediction.graphlet_history).astype(
        np.float64
    )
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
    graphlet_distance = float(np.mean(block_distances)) if block_distances else 0.0
    mass_distance = (
        float(np.mean(np.abs(current_mass - target_mass))) if current_mass.size else 0.0
    )
    return graphlet_distance + float(mass_weight) * mass_distance


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
            preserve_removed_edge_type=cfg.preserve_removed_edge_type,
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
                preserve_removed_edge_type=cfg.preserve_removed_edge_type,
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
                row["hybrid_score"] + cfg.graphlet_weight * row["graphlet_gain"]
            )
            row["candidate_graph"] = candidate_graph
    for row in rows:
        if row["candidate_graph"] is None and np.isfinite(row["hybrid_score"]):
            row["candidate_graph"] = apply_hybrid_action(
                graph,
                row["action"],
                prediction,
                vocabulary,
                preserve_removed_edge_type=cfg.preserve_removed_edge_type,
            )
    if cfg.enforce_molecular_valence or cfg.rdkit_candidate_check:
        for row in rows:
            candidate_graph = row["candidate_graph"]
            if not isinstance(candidate_graph, nx.Graph):
                continue
            molecular_valid = True
            if cfg.enforce_molecular_valence:
                molecular_valid = is_molecular_valence_feasible(
                    candidate_graph,
                    allowed_atom_types=vocabulary.node_values,
                    allowed_bond_types=cfg.molecular_allowed_bond_types,
                )
            if molecular_valid and cfg.rdkit_candidate_check:
                from grapher.molecular.graph_io import is_valid_molecular_graph

                molecular_valid = is_valid_molecular_graph(candidate_graph)
            row["molecular_valid"] = bool(molecular_valid)
            if not molecular_valid:
                row["hybrid_score"] = float("-inf")
    return rows


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
        elif cfg.require_same_edge_type_pair:
            candidates = _sample_same_edge_type_actions(
                current,
                int(cfg.candidate_budget),
                generator,
                vocabulary,
                preserve_connectivity=cfg.preserve_connectivity,
                attempt_multiplier=cfg.molecular_candidate_attempt_multiplier,
            )
        else:
            candidates = sample_valid_double_edge_swaps(
                current,
                int(cfg.candidate_budget),
                generator,
                preserve_connectivity=cfg.preserve_connectivity,
            )
        if cfg.require_same_edge_type_pair:
            candidates = [
                action
                for action in candidates
                if _removed_edge_category(current, action, vocabulary) is not None
            ]
            if cfg.candidate_budget > 0:
                candidates = candidates[: int(cfg.candidate_budget)]
        if not candidates:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": (
                        "no_same_type_candidates"
                        if cfg.require_same_edge_type_pair
                        else "no_candidates"
                    ),
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
        finite = [row for row in rows if np.isfinite(float(row["hybrid_score"]))]
        chosen = None
        if finite:
            scores = np.asarray(
                [float(row["hybrid_score"]) for row in finite],
                dtype=np.float64,
            )
            if cfg.selection in {"greedy", "argmax"}:
                chosen = finite[int(np.argmax(scores))]
            else:
                logits = scores / max(float(cfg.temperature), 1.0e-12)
                logits -= float(np.max(logits))
                probabilities = np.exp(logits)
                probabilities /= float(probabilities.sum())
                chosen = finite[int(generator.choice(len(finite), p=probabilities))]
        rejected_molecular = sum(row.get("molecular_valid") is False for row in rows)
        if chosen is None:
            trace.append(
                {
                    "step": step,
                    "accepted": False,
                    "reason": "no_scored_candidate",
                    "sampled_target_degree_match": prediction.sampled_degree_match,
                    "rejected_molecular_candidates": rejected_molecular,
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
                    "rejected_molecular_candidates": rejected_molecular,
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
                "rejected_molecular_candidates": rejected_molecular,
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

    if return_trace:
        return current, trace
    return current
