"""Implementation of saved-endpoint and common-randomness sampling studies."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from grapher.models.grapher import basis_sampling_ablation as basis_control
from grapher.models.grapher import degree_constrained_realization as realization
from grapher.models.grapher.matched_ablation import (
    _connected, _degrees, _json, _plain, _population, _record, _refine_one,
    _require, _same_invariants, _seed, _sha, _write,
)
from grapher.utils.networkx_pickle import load_trusted_networkx_pickle

FORMAT = "graphes_sampling_ablation_v1"


@contextmanager
def measure_reverse(module, device):
    """Count actual attempted/completed reverse batches, including exceptions."""
    original = module.sample_batch
    stats = {"reverse_calls": 0, "attempted_reverse_samples": 0,
             "completed_reverse_samples": 0, "reverse_seconds": 0.0}
    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    def measured(*args, **kwargs):
        count = int(kwargs["donor_adjacencies"].shape[0])
        stats["reverse_calls"] += 1
        stats["attempted_reverse_samples"] += count
        sync(); started = time.monotonic()
        try:
            result = original(*args, **kwargs)
            stats["completed_reverse_samples"] += count
            return result
        finally:
            sync()
            stats["reverse_seconds"] += time.monotonic() - started
    module.sample_batch = measured
    try:
        yield stats
    finally:
        module.sample_batch = original


def decode_temperature(scores, degrees, config, *, seed, temperature):
    """Use one saved degree target and one common per-ID Gumbel draw per arm."""
    matrix = np.asarray(scores, dtype=np.float64)
    n = len(degrees)
    _require(matrix.shape == (n, n) and np.isfinite(matrix).all(), "Invalid saved decoder scores")
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    u, v = realization._edge_index(n)
    normalized = realization._normalize_scores(matrix[u, v], config.get("score_transform", "zscore"),
                                                float(config.get("score_clip", 8.0)))
    options = copy.deepcopy(config)
    options.update(gumbel_temperature=float(temperature), failure_policy="error")
    return realization._solve_exact_degree_realization(np.asarray(degrees, dtype=np.int64), normalized,
                                                       rng=np.random.default_rng(seed), config=options)


def sample_budget(seed, index, minimum, maximum):
    rng = np.random.default_rng(_seed(seed, index, 610001))
    return minimum + int(rng.random() * (maximum - minimum + 1))


def _implementation_hashes():
    here = Path(__file__).resolve().parent
    return {name: _sha(here / name) for name in
            ("sampling_ablation.py", "basis_sampling_ablation.py", "eigenbasis_perturbation.py",
             "attributed_diversification.py", "degree_constrained_realization.py", "graphlet_stage3.py",
             "vanilla_gsdm.py", "hybrid_attributed_loggap.py")}


class Study:
    def __init__(self, args, cfg, generation, context, checkpoint, arms):
        self.args, self.cfg, self.generation = args, cfg, generation
        self.directory, self.root = Path(args.generation_dir), Path(args.output_root)
        self.checkpoint = Path(checkpoint)
        self.ids = [f"{args.seed}:{i}" for i in range(args.num_samples)]
        self.sources = {}
        self.loaded = {}
        self.state = None
        self.started = time.monotonic()
        pairing = ("same checkpoint, node-count schedule, reverse and attribute RNG seeds; fresh endpoints"
                   if args.study.startswith("basis_") else
                   "same parent sample IDs and donor indices; direct replay of actual training graphs"
                   if args.study == "donor_replay" else
                   "same saved parent endpoints and requested sample IDs")
        self.shared = {**copy.deepcopy(context),
                       "source_generation_manifest_sha256": _sha(self.directory / "manifest.json"),
                       "identity_scheme": "verified_parent_list_order",
                       "pairing": pairing,
                       "reverse_diffusion_rerun": args.study.startswith("basis_")}
        self.plan = {"format": FORMAT, "study": args.study, "arms": arms,
                     "shared_context": self.shared, "implementation_sha256": _implementation_hashes(),
                     "basis_parameters": {k: getattr(args, k) for k in
                         ("perturbation_sigma", "rotations", "repeated_atol", "repeated_rtol")},
                     "device": args.device}
        self.signature = basis_control.digest(self.plan)
        self.root.mkdir(parents=True, exist_ok=True)
        intent = self.root / "sampling_ablation_intent.json"
        if intent.exists():
            _require(_json(intent).get("signature") == self.signature,
                     "Existing study has different parents, settings or implementation; use a new output root")
        else:
            _require(not any(self.root.iterdir()), "Output root is nonempty without a sampling intent")
            _write(intent, {"signature": self.signature, **self.plan})
        self.children = {}

    def load(self, name):
        if name not in self.loaded:
            path = _record(self.directory, self.generation.get(name), name)
            self.sources[name] = copy.deepcopy(self.generation[name])
            self.loaded[name] = load_trusted_networkx_pickle(path)
            _require(isinstance(self.loaded[name], list) and len(self.loaded[name]) == len(self.ids),
                     f"Saved {name} has wrong population length")
        return self.loaded[name]

    def load_state(self):
        if self.state is None:
            _require(_sha(self.checkpoint) == self.shared["checkpoint_sha256"], "Checkpoint changed")
            self.state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        return self.state

    def graphs(self, name):
        return _population(self.load(name), self.ids, name)

    def existing(self, name):
        directory = self.root / name
        if not directory.exists():
            return False
        manifest = _json(directory / "manifest.json")
        _require(manifest.get("study_signature") == self.signature, f"Conflicting arm {name}")
        _require(manifest.get("shared_context") == self.shared, f"Changed arm provenance: {name}")
        for key in ("base_graphs", "diagnostics"):
            _record(directory, manifest.get(key), f"{name}.{key}")
        ids = manifest.get("sample_ids", [])
        _require(manifest.get("requested_sample_ids") == self.ids, f"Requested IDs changed: {name}")
        failures = manifest.get("failures", [])
        _require(len(set(ids)) == len(ids) and set(ids).issubset(self.ids), f"Invalid successful IDs: {name}")
        failed_ids = [row["sample_id"] for row in failures]
        _require(len(set(failed_ids)) == len(failed_ids) and not set(ids).intersection(failed_ids)
                 and set(ids) | set(failed_ids) == set(self.ids), f"Incomplete outcome accounting: {name}")
        _require(manifest.get("completed_count") == len(ids) and manifest.get("failed_count") == len(failures),
                 f"Changed outcome counts: {name}")
        _population(load_trusted_networkx_pickle(directory / "base_graphs.pkl"), ids, name, labelled=True)
        self.children[name] = manifest
        return True

    def save(self, arm, graphs, rows, failures, *, elapsed, timing_scope, extra=None):
        name = arm["name"]
        successful = [str(graph.graph["sample_id"]) for graph in graphs]
        _population(graphs, successful, name, labelled=True)
        _require(len(set(successful)) == len(successful), f"Duplicate successful IDs: {name}")
        _require(set(successful).isdisjoint(row["sample_id"] for row in failures), "Success/failure IDs overlap")
        _require(set(successful) | {row["sample_id"] for row in failures} == set(self.ids), "Missing outcome IDs")
        staging = Path(tempfile.mkdtemp(prefix=".arm_", dir=self.root))
        try:
            graph_path = staging / "base_graphs.pkl"
            with graph_path.open("xb") as stream:
                pickle.dump(graphs, stream, protocol=pickle.HIGHEST_PROTOCOL)
            diagnostics = {"per_graph": rows, "per_sample": rows, "failures": failures,
                           "time_seconds": elapsed, "timing_scope": timing_scope,
                           "requested_count": len(self.ids),
                           "attempted_count": (extra or {}).get("attempted_count", len(self.ids)),
                           "completed_count": len(graphs), "failed_count": len(failures)}
            _write(staging / "diagnostics.json", diagnostics)
            manifest = {"format": FORMAT, "variant": name, "label": name,
                        "study": self.args.study, "study_signature": self.signature,
                        "generation_seed": self.args.seed, "parameters": arm,
                        "requested_count": len(self.ids), "attempted_count": len(self.ids),
                        "completed_count": len(graphs), "failed_count": len(failures),
                        "sample_ids": successful, "requested_sample_ids": self.ids,
                        "status": "complete" if not failures else ("partial" if graphs else "failed"),
                        "failures": failures, "shared_context": self.shared, "provenance": self.shared,
                        "source_artifacts": self.sources,
                        "base_graphs": {"path": "base_graphs.pkl", "sha256": _sha(graph_path)},
                        "diagnostics": {"path": "diagnostics.json", "sha256": _sha(staging / "diagnostics.json")},
                        "time_seconds": elapsed, "end_to_end_seconds": None,
                        "timing_scope": timing_scope, "posthoc_repair": False,
                        "replacement_generated": False, **(extra or {})}
            _write(staging / "manifest.json", manifest)
            os.rename(staging, self.root / name)
            self.children[name] = manifest
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def per_sample(self, arm, operation, *, timing_scope):
        if self.existing(arm["name"]):
            return
        graphs, rows, failures = [], [], []
        started = time.monotonic()
        for index, sample_id in enumerate(self.ids):
            tick = time.monotonic()
            try:
                graph, row = operation(index)
                graph.graph["sample_id"] = sample_id
                rows.append({**_plain(row), "sample_id": sample_id, "graph_index": index,
                             "stage_seconds": time.monotonic() - tick})
                graphs.append(graph)
            except Exception as error:
                failures.append({"sample_id": sample_id, "graph_index": index,
                                 "error": f"{type(error).__name__}: {error}",
                                 "stage_seconds": time.monotonic() - tick,
                                 "replacement_generated": False})
            if (index + 1) % 128 == 0 or index + 1 == len(self.ids):
                print(f"{arm['name']}: {index + 1}/{len(self.ids)}, failures={len(failures)}", flush=True)
        self.save(arm, graphs, rows, failures, elapsed=time.monotonic() - started, timing_scope=timing_scope)

    def finish(self):
        _require(_sha(self.directory / "manifest.json") == self.shared["source_generation_manifest_sha256"],
                 "Parent manifest changed during study")
        _require(_sha(self.checkpoint) == self.shared["checkpoint_sha256"], "Checkpoint changed during study")
        for name, record in self.sources.items():
            _record(self.directory, record, name)
        manifest = {"format": FORMAT, "study": self.args.study, "signature": self.signature,
                    "sample_ids": self.ids, "num_requested": len(self.ids),
                    "shared_context": self.shared, "provenance": self.shared,
                    "branches": self.children, "source_artifacts": self.sources,
                    "invocation_seconds": time.monotonic() - self.started}
        _write(self.root / "ablation_manifest.json", manifest)
        failed = sum(child["failed_count"] for child in self.children.values())
        return {"manifest": str(self.root / "ablation_manifest.json"), "arms": list(self.children),
                "failed_count": failed}


def _budgets(study, arms):
    populations = {"T": study.graphs("pre_realization_graphs"), "D": study.graphs("pre_refinement_graphs")}
    predictions = study.load("predicted_structure_summaries")
    state = study.load_state()
    payload = state.get("structure_summary", {}) or {}
    if not payload.get("enabled", False):
        payload = state.get("graphlet_summary", {}) or {}
    _require(payload.get("enabled") is True, "Checkpoint lacks frozen topology-summary metadata")
    for arm in arms:
        def operation(i):
            source = populations[arm["decoder"]][i]
            _require(predictions[i].get("graph_index") == i, "Saved summary index mismatch")
            budget = sample_budget(study.args.seed, i, arm["budget_min"], arm["budget_max"])
            seed = _seed(study.args.seed, i, 620001)
            if budget == 0:
                graph, row = source.copy(), {"requested_steps": 0, "accepted_steps": 0, "stop_reason": "zero_budget"}
            else:
                graph, row = _refine_one(source, predictions[i], payload, study.cfg, seed, i, budget)
            _same_invariants(source, graph, connectivity=True, label=f"{arm['name']}:{i}")
            return graph, {**row, "rewiring_seed": seed, "requested_steps": budget,
                           "changed": set(graph.edges()) != set(source.edges())}
        study.per_sample(arm, operation, timing_scope="saved-endpoint rewiring/copy only; includes failed attempts; no reverse sampling")


def _temperatures(study, arms):
    scores = study.load("continuous_adjacencies")
    degrees = study.load("projected_realization_degree_sequences")
    parents = study.graphs("pre_refinement_graphs")
    config = realization.resolve_config(study.cfg.get("sample", {}))
    _require(not config.get("candidate_selection", {}).get("enabled"), "Temperature sweep requires one unselected MILP draw")
    predictions = study.load("predicted_structure_summaries")
    state = study.load_state()
    payload = state.get("structure_summary", {}) or {}
    if not payload.get("enabled", False):
        payload = state.get("graphlet_summary", {}) or {}
    _require(payload.get("enabled") is True, "Temperature D/DR pairs require frozen topology-summary metadata")
    for i, graph in enumerate(parents):
        _require(_degrees(graph) == [int(d) for d in degrees[i]], f"Saved indexed degrees differ: {i}")
    for arm in arms:
        refined_arm = {**arm, "name": arm["name"].replace("D_", "DR_", 1),
                       "decoder_source": arm["name"]}
        if (study.root / refined_arm["name"]).exists() and not (study.root / arm["name"]).exists():
            raise ValueError("A saved DR temperature arm is missing its exact D endpoint; refusing a substitute redraw")
        def operation(i):
            seed = _seed(study.args.seed, i, int(config.get("seed_offset", 810001)))
            graph, row = decode_temperature(scores[i], degrees[i], config, seed=seed, temperature=arm["temperature"])
            _require(_degrees(graph) == list(degrees[i]), "Decoder changed fixed projected degrees")
            _require(not config.get("require_connected", True) or _connected(graph), "Disconnected exact-degree output")
            return graph, {"gumbel_temperature": arm["temperature"], "gumbel_seed": seed,
                           "requested_steps": 0, "accepted_steps": 0, "solver": _plain(row)}
        study.per_sample(arm, operation, timing_scope="saved-score exact-degree MILP only; includes failed attempts; fixed projected degrees")
        if study.existing(refined_arm["name"]):
            continue
        # Read back the immutable D artifact so DR is literally derived from
        # that saved realization, including on resume after an interrupted run.
        source_dir = study.root / arm["name"]
        decoder_manifest = study.children[arm["name"]]
        decoder_graphs = load_trusted_networkx_pickle(_record(source_dir, decoder_manifest["base_graphs"], "temperature D endpoint"))
        by_id = dict(zip(decoder_manifest["sample_ids"], decoder_graphs))
        decoder_rows = _json(_record(source_dir, decoder_manifest["diagnostics"], "temperature D diagnostics"))
        solver_by_id = {row["sample_id"]: row.get("solver") for row in decoder_rows["per_graph"]}
        failed_by_id = {row["sample_id"]: row for row in decoder_manifest["failures"]}
        graphs, rows, failures = [], [], []
        attempted = 0
        started = time.monotonic()
        for i, sample_id in enumerate(study.ids):
            if sample_id in failed_by_id:
                failures.append({**failed_by_id[sample_id], "inherited_decoder_failure": True,
                                 "rewiring_attempted": False})
                continue
            tick = time.monotonic(); attempted += 1
            seed = _seed(study.args.seed, i, 620001)
            budget = sample_budget(study.args.seed, i, arm.get("rewire_budget_min", 1), arm.get("rewire_budget_max", 8))
            try:
                _require(predictions[i].get("graph_index") == i, "Saved summary index mismatch")
                source = by_id[sample_id]
                if budget == 0:
                    graph, row = source.copy(), {"requested_steps": 0, "accepted_steps": 0, "stop_reason": "zero_budget"}
                else:
                    graph, row = _refine_one(source, predictions[i], payload, study.cfg, seed, i, budget)
                _same_invariants(source, graph, connectivity=True, label=f"{refined_arm['name']}:{i}")
                graph.graph["sample_id"] = sample_id
                graphs.append(graph)
                rows.append({**_plain(row), "sample_id": sample_id, "graph_index": i,
                             "gumbel_temperature": arm["temperature"], "solver": solver_by_id[sample_id],
                             "rewiring_seed": seed, "requested_steps": budget,
                             "stage_seconds": time.monotonic() - tick})
            except Exception as error:
                failures.append({"sample_id": sample_id, "graph_index": i,
                                 "error": f"{type(error).__name__}: {error}", "rewiring_attempted": True,
                                 "stage_seconds": time.monotonic() - tick, "replacement_generated": False})
        elapsed = time.monotonic() - started
        study.save(refined_arm, graphs, rows, failures, elapsed=elapsed,
                   timing_scope="rewiring exact saved D temperature endpoints only; paired decoder cost recorded separately",
                   extra={"attempted_count": attempted,
                          "paired_decoder_artifact": {"path": str(source_dir / "base_graphs.pkl"),
                                                       "sha256": decoder_manifest["base_graphs"]["sha256"]},
                          "time_accounting": {"decoder_seconds": decoder_manifest["time_seconds"],
                                              "rewiring_seconds": elapsed,
                                              "reverse_diffusion_rerun": False}})


def _masks(study, arms):
    from grapher.models.grapher import attributed_diversification as diversification
    from grapher.properties.summary import SummaryConfig
    from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
    from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
    source = study.graphs("pre_refinement_graphs")
    topology = study.load("predicted_topology_summaries")
    typed = study.load("predicted_typed_graphlet_summaries")
    state = study.load_state()
    vocab = GraphCategoryVocabulary.from_dict(state["vocabulary"])
    top_basis = TopologyGraphletBasis.from_dict(state["topology_summary"]["graphlet_basis"])
    typed_basis = GraphletBasis.from_dict(state["attribute_summary"]["graphlet_basis"])
    top_summary = SummaryConfig.from_dict(state["topology_summary"]["summary_config"])
    typed_summary = SummaryConfig.from_dict(state["attribute_summary"]["summary_config"])
    options = diversification.resolve_options(study.cfg["attributed_diversification"])
    # Derive budgets from an isolated per-ID stream, identically in both arms.
    budgets = [diversification.sample_step_budget(np.random.default_rng(_seed(study.args.seed, i, 610001)), options)
               for i in range(len(study.ids))]
    for arm in arms:
        def operation(i):
            _require(topology[i].get("graph_index") == i and typed[i].get("graph_index") == i,
                     "Saved molecular summary index mismatch")
            local = copy.deepcopy(options)
            local["preserve_neighbor_atom_type_by_bond_type"] = arm["strict_neighbor_mask"]
            local["random_steps"] = {"distribution": "uniform_integer", "min_steps": budgets[i], "max_steps": budgets[i]}
            seed = _seed(study.args.seed, i, 620001)
            graph, row = diversification.diversify_same_bond_graph(
                source[i], topology_prediction=topology[i], attribute_prediction=typed[i],
                topology_basis=top_basis, typed_basis=typed_basis, topology_summary_config=top_summary,
                typed_summary_config=typed_summary, vocabulary=vocab, config=local,
                rng=np.random.default_rng(seed))
            _same_invariants(source[i], graph, connectivity=True, label=f"{arm['name']}:{i}")
            return graph, {**row, "planned_budget": budgets[i], "rewiring_seed": seed,
                           "strict_neighbor_mask": arm["strict_neighbor_mask"]}
        study.per_sample(arm, operation, timing_scope="same saved attributed endpoints and frozen summaries; molecular rewiring only")


def _replay(study, arms):
    state = study.load_state()
    indices = study.load("sampled_basis_indices")
    sizes = np.asarray(state["basis_num_nodes"])
    bank = np.asarray(state["basis_adjacencies"])
    training_path = study.args.dataset_root / study.args.serialized_dataset / "train.pkl"
    if study.args.kind == "molecular":
        from grapher.models.grapher.hierarchical_attributed_loggap import _attributed_graphs
        training = _attributed_graphs(training_path)
    else:
        from grapher.models.grapher.vanilla_gsdm import _graphs
        training = _graphs(training_path)
    _require(len(training) == len(bank), "Training graph order cannot be matched to checkpoint donor bank")
    def operation(i):
        donor = int(indices[i]); n = int(sizes[donor]); graph = training[donor].copy()
        observed = nx.to_numpy_array(graph, nodelist=range(n), weight=None)
        _require(observed.shape == bank[donor, :n, :n].shape and np.array_equal(observed, bank[donor, :n, :n]),
                 f"Training donor adjacency differs from checkpoint: {donor}")
        return graph, {"original_donor_index": donor, "training_donor_replay": True,
                       "requested_steps": 0, "accepted_steps": 0}
    study.per_sample(arms[0], operation, timing_scope="copy actual matched training donors; no neural generation; memorization diagnostic")


def _build_basis_refiner(study, directory, generation):
    """Freeze new endpoint predictions and isolate per-ID postprocessing RNG."""
    state = study.load_state()
    def read(name):
        value = load_trusted_networkx_pickle(_record(directory, generation.get(name), name))
        _require(isinstance(value, list) and len(value) == len(study.ids), f"Invalid basis {name} population")
        _require(all(row.get("graph_index") == i for i, row in enumerate(value)), f"Invalid basis {name} order")
        return value
    if study.args.kind == "matched":
        from grapher.models.grapher.graphlet_stage3 import _sample_random_step_budget
        predictions = read("predicted_structure_summaries")
        payload = state.get("structure_summary", {}) or {}
        if not payload.get("enabled", False):
            payload = state.get("graphlet_summary", {}) or {}
        _require(payload.get("enabled") is True, "Basis TR requires frozen topology-summary metadata")
        cfg = study.cfg["graphlet_refinement"]
        _require(cfg.get("enabled") is True, "Basis TR requires the configured main rewiring operator")
        def refine(i, source):
            budget = _sample_random_step_budget(np.random.default_rng(_seed(study.args.seed, i, 610001)), cfg,
                                               fallback=int(cfg.get("refiner", {}).get("steps", 0)))
            seed = _seed(study.args.seed, i, 620001)
            graph, row = _refine_one(source, predictions[i], payload, study.cfg, seed, i, budget)
            return graph, {**row, "planned_budget": budget, "rewiring_seed": seed}
        return refine
    from grapher.models.grapher import attributed_diversification as diversification
    from grapher.properties.summary import SummaryConfig
    from grapher.rewiring_mlp.attributed.data import GraphCategoryVocabulary, GraphletBasis
    from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
    topology, typed = read("predicted_topology_summaries"), read("predicted_typed_graphlet_summaries")
    vocab = GraphCategoryVocabulary.from_dict(state["vocabulary"])
    top_basis = TopologyGraphletBasis.from_dict(state["topology_summary"]["graphlet_basis"])
    typed_basis = GraphletBasis.from_dict(state["attribute_summary"]["graphlet_basis"])
    top_summary = SummaryConfig.from_dict(state["topology_summary"]["summary_config"])
    typed_summary = SummaryConfig.from_dict(state["attribute_summary"]["summary_config"])
    cfg = diversification.resolve_options(study.cfg["attributed_diversification"])
    _require(cfg.get("enabled") is True, "Basis molecular TR requires configured main rewiring")
    def refine(i, source):
        local = copy.deepcopy(cfg)
        budget = diversification.sample_step_budget(np.random.default_rng(_seed(study.args.seed, i, 610001)), cfg)
        local["random_steps"] = {"distribution": "uniform_integer", "min_steps": budget, "max_steps": budget}
        seed = _seed(study.args.seed, i, 620001)
        graph, row = diversification.diversify_same_bond_graph(
            source, topology_prediction=topology[i], attribute_prediction=typed[i], topology_basis=top_basis,
            typed_basis=typed_basis, topology_summary_config=top_summary, typed_summary_config=typed_summary,
            vocabulary=vocab, config=local, rng=np.random.default_rng(seed))
        return graph, {**row, "planned_budget": budget, "rewiring_seed": seed}
    return refine


def _basis(study, arms):
    from grapher.models.base import GenerateRequest, RunSpec
    from grapher.models.grapher.wrapper import GraphERWrapper
    if study.args.kind == "molecular":
        from grapher.models.grapher import hybrid_attributed_loggap as generator_module
    else:
        from grapher.models.grapher import vanilla_gsdm as generator_module
    state = study.load_state()
    parents = study.load("sampled_basis_indices")
    sizes = np.asarray(state["basis_num_nodes"])
    expected_sizes = [int(sizes[int(i)]) for i in parents]
    run_dir = study.args.run_dir.resolve()
    run = RunSpec("grapher", study.args.dataset, run_dir.name, study.args.seed, output_root=run_dir.parents[2])
    _require(run.layout.run_dir.resolve() == run_dir, "Unexpected managed run-directory layout")
    for arm in arms:
        branches = [dict(arm, name=arm["name"] + suffix) for suffix in ("_T", "_TR")]
        if all(study.existing(branch["name"]) for branch in branches):
            continue
        control = arm["control"]
        cap = arm.get("cap_per_size")
        offset = (1 if control == "reservoir" else arm.get("donor_offset", 0))
        schedule = basis_control.make_schedule(parents, sizes, seed=study.args.seed,
                                               cap_per_size=cap, donor_offset=offset)
        sample = copy.deepcopy(study.cfg["sample"])
        sample["basis_ablation"] = schedule
        sample["matched_ablation"] = {"enabled": False}
        sample.setdefault("degree_constrained_realization", {})["enabled"] = False
        perturbation = {"enabled": control in {"sign", "repeated", "random_rotation"},
                        "stage": "reverse", "method": "sign" if control == "sign" else "givens",
                        "pairing": "repeated" if control == "repeated" else "random",
                        "sigma": study.args.perturbation_sigma, "num_rotations": study.args.rotations,
                        "repeated_atol": study.args.repeated_atol, "repeated_rtol": study.args.repeated_rtol}
        sample["eigenbasis_perturbation"] = perturbation
        options = {"runtime": {"device": study.args.device}, "sample": sample,
                   "generation_batch_size": study.cfg["generation_batch_size"]}
        for key in ("graphlet_refinement", "attributed_diversification"):
            if key in study.cfg:
                options[key] = copy.deepcopy(study.cfg[key])
                options[key]["enabled"] = False
        if study.args.kind == "molecular":
            _require(sample.get("separate_attribute_rng", True), "Basis controls require separate molecular attribute RNG")
        descriptor = {"study_signature": study.signature, "arm": arm, "options": options,
                      "checkpoint_sha256": study.shared["checkpoint_sha256"]}
        generation_id = "sampling_" + basis_control.digest(descriptor)[:20] + "_" + arm["name"]
        target = run.layout.generation_dir(generation_id)
        intent_path = target.parent / (generation_id + ".sampling_intent.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if intent_path.exists():
            _require(_json(intent_path) == _plain(descriptor), "Basis generation intent differs")
        else:
            _require(not target.exists(), "Unverified basis generation directory already exists")
            _write(intent_path, descriptor)
        tick = time.monotonic()
        new_generation = not target.exists()
        stats = None
        if new_generation:
            try:
                with measure_reverse(generator_module, study.args.device) as stats:
                    GraphERWrapper().generate(GenerateRequest(run, study.checkpoint, len(study.ids), study.args.seed,
                                                             generation_id=generation_id, options=options))
                    if str(study.args.device).startswith("cuda"):
                        torch.cuda.synchronize()
            except Exception as error:
                elapsed = time.monotonic() - tick
                attempted = stats["attempted_reverse_samples"] if stats is not None else 0
                failures = [{"sample_id": sample_id, "graph_index": i,
                             "error": f"Generation aborted: {type(error).__name__}: {error}",
                             "reverse_attempted": i < attempted, "replacement_generated": False}
                            for i, sample_id in enumerate(study.ids)]
                for branch in branches:
                    if not study.existing(branch["name"]):
                        study.save(branch, [], [], failures, elapsed=elapsed,
                                   timing_scope="aborted reverse-generation invocation; unattempted IDs are explicitly marked",
                                   extra={"attempted_count": attempted, "reverse_measurement": stats,
                                          "end_to_end_seconds": elapsed, "basis_schedule": schedule})
                continue
        generation = _json(target / "manifest.json")
        _require(generation.get("checkpoint", {}).get("sha256") == study.shared["checkpoint_sha256"],
                 "Basis generation checkpoint differs")
        _require(generation.get("num_generated") == len(study.ids) and generation.get("generation_seed") == study.args.seed,
                 "Basis generation count/seed mismatch")
        effective = generation.get("effective_sample_config", {})
        _require(effective.get("basis_ablation") == _plain(schedule), "Saved basis schedule differs")
        index_path = _record(target, generation.get("sampled_basis_indices"), "basis generation donor indices")
        _require(load_trusted_networkx_pickle(index_path) == schedule["indices"], "Applied donor schedule differs")
        elapsed = time.monotonic() - tick if new_generation else generation.get("duration_seconds")
        measurement_path = target.parent / (generation_id + ".sampling_measurement.json")
        if new_generation:
            _write(measurement_path, {"generation_manifest_sha256": _sha(target / "manifest.json"),
                                      "end_to_end_seconds": elapsed, "reverse_measurement": stats})
        elif measurement_path.exists():
            measured = _json(measurement_path)
            _require(measured.get("generation_manifest_sha256") == _sha(target / "manifest.json"),
                     "Basis timing receipt belongs to a different generation")
            elapsed, stats = measured["end_to_end_seconds"], measured["reverse_measurement"]
        basis_rows = []
        if generation.get("eigenbasis_perturbation_diagnostics"):
            diag_path = _record(target, generation["eigenbasis_perturbation_diagnostics"], "basis diagnostics")
            basis_rows = _json(diag_path).get("per_graph", [])
        graphs = load_trusted_networkx_pickle(_record(target, generation.get("base_graphs"), "basis T endpoints"))
        _population(graphs, study.ids, "basis T endpoints")
        common_extra = {"timing_shared_run_id": generation_id, "reverse_measurement": stats,
                        "native_generation_manifest": {"path": str(target / "manifest.json"), "sha256": _sha(target / "manifest.json")},
                        "basis_schedule": schedule, "basis_perturbation": perturbation,
                        "postprocessing_rng": "isolated per-ID budgets and rewiring streams, shared across all basis arms"}
        refiner = None
        for branch in branches:
            if study.existing(branch["name"]):
                continue
            do_rewire = branch["name"].endswith("_TR")
            if do_rewire and refiner is None:
                refiner = _build_basis_refiner(study, target, generation)
            rows, failures, outputs = [], [], []
            rewire_start = time.monotonic()
            for i, source in enumerate(graphs):
                tick = time.monotonic()
                try:
                    _require(len(source) == expected_sizes[i], "Basis study changed paired node-count schedule")
                    graph, row = refiner(i, source) if do_rewire else (source.copy(), {"requested_steps": 0, "accepted_steps": 0})
                    _same_invariants(source, graph, connectivity=True, label=f"{branch['name']}:{i}")
                    graph.graph["sample_id"] = study.ids[i]
                    outputs.append(graph)
                    rows.append({**_plain(row), "sample_id": study.ids[i], "graph_index": i,
                                 "original_donor_index": schedule["indices"][i], "num_nodes": len(graph),
                                 "stage_seconds": time.monotonic() - tick if do_rewire else None,
                                 "basis_perturbation": basis_rows[i] if basis_rows else None})
                except Exception as error:
                    failures.append({"sample_id": study.ids[i], "graph_index": i,
                                     "error": f"{type(error).__name__}: {error}",
                                     "stage_seconds": time.monotonic() - tick, "replacement_generated": False})
            rewire_elapsed = time.monotonic() - rewire_start if do_rewire else 0.0
            study.save(branch, outputs, rows, failures, elapsed=rewire_elapsed if do_rewire else elapsed,
                       timing_scope=("rewiring saved basis-specific T endpoints only; reverse-generation cost recorded separately"
                                     if do_rewire else "fresh reverse generation, threshold and attributes if molecular; no rewiring"),
                       extra={**common_extra, "end_to_end_seconds": elapsed + rewire_elapsed if elapsed is not None else None,
                              "time_accounting": {"native_generation_seconds": elapsed,
                                                  "rewiring_seconds": rewire_elapsed}})


def run_study(args, cfg, generation, context, checkpoint, arms):
    study = Study(args, cfg, generation, context, checkpoint, arms)
    handlers = {"rewiring_budget": _budgets, "gumbel_temperature": _temperatures,
                "molecular_masks": _masks, "donor_replay": _replay,
                "basis_reservoir": _basis, "basis_sensitivity": _basis}
    handlers[args.study](study, arms)
    return study.finish()
