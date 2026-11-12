#!/usr/bin/env python3
"""Measured rewiring drift and faithful replay of saved edge-swap traces.

Inputs are hash-anchored exported branches from the same matched generation.
Only trusted project graph/diagnostic pickles may be loaded. No model is fit,
no rejected graph is replaced, and no missing intermediate state is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import analyze_grapher_paper_experiments as evidence
from evaluate_grapher_prediction_diagnostics import diagnostic_index, plain

FORMAT = "graphes_rewiring_evidence_v1"


def load_pair(before_branch: Path, after_branch: Path):
    before, bmeta, _bcommon, bpath = evidence.checked_population(before_branch.resolve())
    after, ameta, _acommon, apath = evidence.checked_population(after_branch.resolve())
    if not bmeta.get("shared_context") or bmeta.get("shared_context") != ameta.get("shared_context"):
        raise ValueError("Branches do not share a nonempty verified generation context")
    if bmeta.get("requested_sample_ids") != ameta.get("requested_sample_ids"):
        raise ValueError("Branches do not share the same ordered requested sample IDs")
    sources = {name: {"path": str(path), "sha256": evidence.sha256(path),
                       "manifest_path": str(branch / "manifest.json"),
                       "manifest_sha256": evidence.sha256(branch / "manifest.json")}
               for name, branch, path in (("before", before_branch.resolve(), bpath),
                                         ("after", after_branch.resolve(), apath))}
    return before, after, bmeta, ameta, sources


def load_diagnostics(after_branch: Path, metadata: dict, explicit: Path | None = None):
    """Accept a file only through its producer's path+SHA-256 declaration."""
    candidates = []
    if isinstance(metadata.get("diagnostics"), dict):
        candidates.append((after_branch, metadata["diagnostics"], "branch"))
    for directory in list(after_branch.parents)[:3]:
        path = directory / "manifest.json"
        if not path.is_file():
            continue
        manifest = evidence.read_json(path)
        for key in ("attributed_diversification_diagnostics", "graphlet_refinement_diagnostics"):
            if isinstance(manifest.get(key), dict):
                candidates.append((directory, manifest[key], key))
    selected = None
    for directory, record, role in candidates:
        candidate = (directory / record.get("path", "")).resolve()
        if explicit is None or candidate == explicit.resolve():
            selected = (directory, record, role)
            break
    if selected is None:
        if explicit is not None:
            raise ValueError("--diagnostics must match a hash-anchored branch/parent diagnostic artifact")
        return {}, None
    directory, record, role = selected
    path = evidence.declared_artifact(directory, record)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix in {".pkl", ".pickle"}:
        from grapher.utils.networkx_pickle import load_trusted_networkx_pickle
        payload = load_trusted_networkx_pickle(path)
    else:
        raise ValueError(f"Unsupported diagnostic artifact: {path}")
    rows = payload.get("per_graph", payload.get("per_sample", [])) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected a list of per-graph diagnostic objects")
    ids = metadata.get("requested_sample_ids")
    if not isinstance(ids, list):
        raise ValueError("Diagnostic binding requires explicit requested sample IDs")
    indexed = diagnostic_index(rows, ids)
    return indexed, {"path": str(path), "sha256": evidence.sha256(path), "role": role,
                     "rows": len(rows), "producer_manifest": str(directory / "manifest.json")}


def topology(graph):
    import networkx as nx
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError("Rewiring evidence requires simple undirected loop-free graphs")
    result = nx.Graph()
    result.add_nodes_from(sorted(graph.nodes(), key=lambda value: (type(value).__name__, repr(value))))
    result.add_edges_from(graph.edges())
    return result


def trace_statistics(row: dict | None) -> dict:
    """Retain measured denominators, including zero and skipped cases."""
    if row is None:
        return {"diagnostics_available": False, "requested_steps": None, "accepted_steps": None,
                "proposals": None, "eligible_candidates_recorded": None, "stop_reason": None,
                "source_eligible_recorded": None, "trace": None}
    trace = row.get("trace")
    trace = trace if isinstance(trace, list) else None
    proposals = None
    proposal_source = None
    totals = row.get("candidate_search_totals", {})
    if isinstance(totals, dict) and evidence.nonnegative_int(totals.get("proposals")) is not None:
        proposals = totals["proposals"]
        proposal_source = "candidate_search_totals.proposals"
    measured_proposals = []
    eligibility = []
    if trace is not None:
        for event in trace:
            count = event.get("num_proposals", event.get("candidate_search", {}).get("proposals"))
            if evidence.nonnegative_int(count) is not None:
                measured_proposals.append(count)
            count = event.get("num_eligible_candidates", event.get("candidates_inside_trust_region"))
            if evidence.nonnegative_int(count) is not None:
                eligibility.append(count)
        if proposals is None and trace and len(measured_proposals) == len(trace):
            proposals = sum(measured_proposals)
            proposal_source = "sum of all recorded decisions' proposal counts"
    stop = row.get("stop_reason", row.get("skip_reason"))
    stop_source = "per-graph diagnostic" if stop is not None else None
    if stop is None and trace:
        terminal = [event for event in trace if event.get("terminal_stop") is True]
        if terminal:
            stop, stop_source = terminal[-1].get("reason"), "last recorded terminal decision"
        elif (evidence.nonnegative_int(row.get("requested_steps")) is not None
              and row.get("accepted_steps") == row["requested_steps"]):
            stop, stop_source = "requested_budget_completed", "derived from recorded accepted_steps == requested_steps"
    eligible = None
    if isinstance(row.get("skipped"), bool):
        eligible = not row["skipped"]
    elif stop == "disconnected_source_skipped":
        eligible = False
    return {"diagnostics_available": True, "requested_steps": row.get("requested_steps"),
            "accepted_steps": row.get("accepted_steps"), "proposals": proposals,
            "proposal_source": proposal_source, "recorded_proposal_sum": sum(measured_proposals) if measured_proposals else None,
            "proposal_decisions_measured": len(measured_proposals) if trace is not None else None,
            "trace_decisions": len(trace) if trace is not None else None,
            "eligible_candidates_recorded": sum(eligibility) if eligibility else None,
            "eligibility_decisions_measured": len(eligibility) if trace is not None else None,
            "source_eligible_recorded": eligible, "stop_reason": stop, "stop_reason_source": stop_source,
            "producer_diagnostic": plain(row), "trace": plain(trace)}


def graphlet_descriptor(graph, basis, config, *, backend: str, samples: int | None, seed: int):
    import numpy as np
    if backend == "exact":
        from grapher.rewiring_mlp.generic.graphlets import extract_topology_graphlet_target
        return extract_topology_graphlet_target(graph, graphlet_basis=basis, summary_config=config)
    from grapher.utils.motifs import graphlet_count_dict
    history, masses = {}, {}
    for order in basis.sizes:
        k = int(order)
        counts = graphlet_count_dict(graph, k, connected_only=True, topology_filter="all",
                                     num_samples=samples, rng=np.random.default_rng(seed + 1009*k))
        total = sum(counts.values())
        history[order] = {key: value / total for key, value in counts.items()} if total else {}
        denominator = min(math.comb(len(graph),k),samples) if len(graph)>=k else 0
        masses[order] = total / denominator if denominator else 0.
    return basis.flatten_history(history).astype(float), basis.flatten_mass(masses).astype(float)


def descriptors(graph, basis, config, args, sample_seed):
    import numpy as np
    import networkx as nx
    from grapher.rewiring_mlp.properties.summary import clustering_histogram, spectral_histogram, python_orbit_count_vector
    graph = topology(graph)
    if len(graph) == 0:
        raise ValueError("Descriptor drift requires nonempty graphs")
    adjacency = nx.to_numpy_array(graph, weight=None, dtype=float)
    eigenvalues = np.linalg.eigvalsh(np.diag(adjacency.sum(axis=1))-adjacency)
    orbits = python_orbit_count_vector(graph)
    orbit_total = float(orbits.sum())
    histogram, mass = graphlet_descriptor(graph,basis,config,backend=args.graphlet_backend,
                                          samples=args.graphlet_samples,seed=sample_seed)
    return {"spectrum":eigenvalues, "spectral_histogram":spectral_histogram(graph,bins=args.spectral_bins),
            "clustering_histogram":clustering_histogram(graph,bins=args.clustering_bins),
            "mean_clustering":nx.average_clustering(graph), "orbit_counts":orbits,
            "orbit_histogram":orbits/orbit_total if orbit_total else np.zeros_like(orbits),
            "graphlet_histogram":histogram,"graphlet_mass":mass}


def paired_drift(before, after, basis, config, args, sample_id):
    import numpy as np
    if set(before) != set(after):
        raise ValueError(f"Indexed node sets differ for {sample_id}")
    seed = int.from_bytes(hashlib.sha256(f"{args.graphlet_seed}:{sample_id}".encode()).digest()[:8],"little")
    a = descriptors(before,basis,config,args,seed)
    b = descriptors(after,basis,config,args,seed)
    result = {
        "laplacian_spectrum_rmse":float(np.sqrt(np.mean((a["spectrum"]-b["spectrum"])**2))),
        "normalized_laplacian_histogram_tv":float(.5*np.abs(a["spectral_histogram"]-b["spectral_histogram"]).sum()),
        "clustering_histogram_tv":float(.5*np.abs(a["clustering_histogram"]-b["clustering_histogram"]).sum()),
        "mean_clustering_absolute_change":float(abs(a["mean_clustering"]-b["mean_clustering"])),
        "orbit_histogram_tv":float(.5*np.abs(a["orbit_histogram"]-b["orbit_histogram"]).sum()),
        "orbit_log1p_count_rmse":float(np.sqrt(np.mean((np.log1p(a["orbit_counts"])-np.log1p(b["orbit_counts"]))**2))),
    }
    graphlet_tvs=[]
    for index,(order,(start,stop)) in enumerate(zip(basis.sizes,basis.slices)):
        value=float(.5*np.abs(a["graphlet_histogram"][start:stop]-b["graphlet_histogram"][start:stop]).sum())
        result[f"graphlet_k{order}_histogram_tv"] = value
        result[f"graphlet_k{order}_mass_absolute_change"] = float(abs(a["graphlet_mass"][index]-b["graphlet_mass"][index]))
        graphlet_tvs.append(value)
    result["graphlet_mean_histogram_tv"] = statistics.fmean(graphlet_tvs)
    edges_a,edges_b={frozenset(edge) for edge in before.edges()},{frozenset(edge) for edge in after.edges()}
    result["normalized_edge_symmetric_difference"] = len(edges_a ^ edges_b)/max(len(edges_a),1)
    return result, {"before":plain(a),"after":plain(b)}


def distribution(values):
    measured=[float(value) for value in values if evidence.saved._finite(value) is not None]
    return {"n_measured":len(measured),"n_missing":len(values)-len(measured),
            "mean":statistics.fmean(measured) if measured else None,
            "std":statistics.stdev(measured) if len(measured)>1 else None,
            "min":min(measured) if measured else None,"max":max(measured) if measured else None,
            **{f"q{round(100*q):02d}":evidence.quantile(measured,q) for q in (.1,.25,.5,.75,.9,.95)},
            "values":measured}


def drift_command(args):
    from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
    from grapher.rewiring_mlp.properties.summary import SummaryConfig
    before,after,bmeta,ameta,sources=load_pair(args.before_branch,args.after_branch)
    diagnostics,diagnostic_source=load_diagnostics(args.after_branch.resolve(),ameta,args.diagnostics)
    config=SummaryConfig.from_dict({"graphlet_history":True,"graphlet_k_min":3,"graphlet_k_max":args.max_graphlet_order,
                                    "graphlet_connected_only":True,"graphlet_topology_filter":"all","graphlet_backend":"exact"})
    basis=TopologyGraphletBasis.from_config(config)
    paired_ids=[sample_id for sample_id in bmeta["requested_sample_ids"] if sample_id in before and sample_id in after]
    rows=[]
    for sample_id in paired_ids:
        values,source_descriptors=paired_drift(before[sample_id],after[sample_id],basis,config,args,sample_id)
        rows.append({"sample_id":sample_id,"metrics":values,"descriptors":source_descriptors,
                     "rewiring":trace_statistics(diagnostics.get(sample_id))})
    metric_names=sorted({name for row in rows for name in row["metrics"]})
    summaries={name:distribution([row["metrics"].get(name) for row in rows]) for name in metric_names}
    rewiring_names=("requested_steps","accepted_steps","proposals","eligible_candidates_recorded")
    rewiring_summaries={name:distribution([row["rewiring"].get(name) for row in rows]) for name in rewiring_names}
    stop_counts=Counter(row["rewiring"].get("stop_reason") for row in rows if row["rewiring"].get("stop_reason") is not None)
    source_invariants=evidence.invariant_pair(before,after,node_attribute=args.node_attribute,edge_attribute=args.edge_attribute)
    result={"format":FORMAT,"kind":"paired_structural_drift","source_files":sources,"diagnostic_source":diagnostic_source,
            "shared_context":bmeta["shared_context"],"paired_ids":paired_ids,
            "requested_count":len(bmeta["requested_sample_ids"]),"paired_count":len(rows),
            "unpaired_requested_ids":[i for i in bmeta["requested_sample_ids"] if i not in set(paired_ids)],
            "before_failures":bmeta.get("failures"),"after_failures":ameta.get("failures"),
            "protocol":{"topology_only_descriptors":True,"attributes_ignored_by_descriptors":True,
                        "provenance_scope":"branch/common-index consistency, graph hashes, requested/successful IDs and shared context; training checkpoint/configuration checks remain the separate check_grapher_ablation_artifacts.py gate",
                        "graphlet_backend":args.graphlet_backend,"graphlet_orders":list(range(3,args.max_graphlet_order+1)),
                        "graphlet_samples_per_order":args.graphlet_samples,"graphlet_seed":args.graphlet_seed,
                        "graphlet_sampling":"same indexed subsets for before/after; seed derived from explicit seed + sample ID + order; exhaustive if choose(n,k)<=sample count" if args.graphlet_backend=="sampled" else "exact connected induced graphlets using existing extract_topology_graphlet_target",
                        "orbit_counting":"existing exact python_orbit_count_vector, 15 mean-per-node ORCA roles for orders 2..4; never sampled",
                        "clustering_bins":args.clustering_bins,"spectral_bins":args.spectral_bins,
                        "spectrum":"RMSE of sorted unweighted combinatorial-Laplacian eigenvalues; histogram TV separately uses normalized Laplacian",
                        "histogram_distance":"0.5*L1; graphlet empty-order blocks remain all zero, so TV is a descriptor change and not always a probability-distance between nonempty distributions",
                        "graphlet_mass":"connected induced subset frequency, reported separately from conditional graphlet histograms",
                        "inference_scope":"paired successful endpoints; distributions describe graphs within this run, not training-seed standard errors",
                        "missing_diagnostics":"null; absent proposal/eligibility fields are not zero; recorded partial totals retain their denominators"},
            "drift_distributions":summaries,"rewiring_distributions":rewiring_summaries,
            "stop_reason_counts":dict(stop_counts),"missing_stop_reason_count":len(rows)-sum(stop_counts.values()),
            "invariants":source_invariants,"per_sample":rows}
    evidence.write_json(args.output_dir/"rewiring_drift_report.json",result)
    evidence.write_csv(args.output_dir/"rewiring_drift_per_sample.csv",[{"sample_id":r["sample_id"],**r["metrics"],
                       **{key:r["rewiring"].get(key) for key in (*rewiring_names,"stop_reason","source_eligible_recorded")}} for r in rows])
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys=("laplacian_spectrum_rmse","normalized_laplacian_histogram_tv","clustering_histogram_tv",
              "orbit_histogram_tv","graphlet_mean_histogram_tv","normalized_edge_symmetric_difference")
        fig,axes=plt.subplots(2,3,figsize=(11,6),layout="constrained")
        for ax,key in zip(axes.flat,keys):
            values=sorted(summaries.get(key,{}).get("values",[]))
            if values:
                ax.step(values,[(i+1)/len(values) for i in range(len(values))],where="post")
            ax.set(title=key.replace("_"," "),ylabel="Empirical cumulative fraction",ylim=(0,1.02))
            ax.grid(alpha=.2)
        fig.savefig(args.output_dir/"rewiring_drift_ecdf.pdf",bbox_inches="tight")
        fig.savefig(args.output_dir/"rewiring_drift_ecdf.png",dpi=180,bbox_inches="tight")
        plt.close(fig)
    print(f"Measured structural drift on {len(rows)} paired IDs; {len(diagnostics)} recorded per-graph diagnostics available.")
    return 0


def graph_signature(graph):
    """Exact indexed topology+attributes; ignore producer graph-level metadata."""
    nodes={str(node):plain(dict(graph.nodes[node])) for node in graph}
    edges=sorted(((repr(min(u,v)),repr(max(u,v)),plain(dict(data))) for u,v,data in graph.edges(data=True)),key=lambda row:row[:2])
    return json.dumps({"nodes":nodes,"edges":edges},sort_keys=True)


def replay_trace(before, after, row: dict, *, kind: str, edge_attribute: str):
    import networkx as nx
    topology(before); topology(after)
    trace=row.get("trace")
    if not isinstance(trace,list):
        raise ValueError("Missing saved trace: endpoint graphs cannot reconstruct unrecorded intermediate states")
    expected=evidence.nonnegative_int(row.get("accepted_steps"))
    if expected is None:
        raise ValueError("Trace replay requires an explicit accepted_steps count")
    states=[before.copy()]
    events=[]
    current=before.copy()
    baseline_degrees=dict(before.degree())
    for trace_index,event in enumerate(trace):
        accepted=event.get("accepted") if kind=="generic" else ("removed" in event and "added" in event)
        if accepted is False:
            events.append({"trace_index":trace_index,"applied":False,"record":plain(event)})
            continue
        if accepted is not True:
            raise ValueError(f"Trace event {trace_index} has no explicit accepted action")
        action=event.get("action") if kind=="generic" else [event["removed"],event["added"]]
        if not isinstance(action,(list,tuple)) or len(action)!=2:
            raise ValueError(f"Malformed saved action at trace event {trace_index}")
        removed,added=([tuple(edge) for edge in part] for part in action)
        if any(len(edges)!=2 or any(len(edge)!=2 for edge in edges) for edges in (removed,added)):
            raise ValueError("A saved double-edge swap must contain two removed and two added edges")
        if (len({frozenset(edge) for edge in removed})!=2 or len({frozenset(edge) for edge in added})!=2
                or any(u==v or u not in current or v not in current for u,v in removed+added)
                or any(not current.has_edge(*edge) for edge in removed)
                or any(current.has_edge(*edge) for edge in added)):
            raise ValueError(f"Recorded swap is not applicable at trace event {trace_index}")
        payload={}
        if kind=="molecular":
            payloads=[plain(dict(current.edges[edge])) for edge in removed]
            if any(edge_attribute not in data for data in payloads):
                raise ValueError("Molecular trace is missing a removed edge's bond category")
            if payloads[0]!=payloads[1]:
                raise ValueError("Molecular trace lacks edge-payload assignment; unequal removed edge attributes make replay ambiguous")
            payload=copy.deepcopy(dict(current.edges[removed[0]]))
        current.remove_edges_from(removed)
        for edge in added:
            current.add_edge(*edge,**copy.deepcopy(payload))
        if dict(current.degree())!=baseline_degrees:
            raise ValueError(f"Recorded swap violates indexed degrees at trace event {trace_index}")
        states.append(current.copy())
        events.append({"trace_index":trace_index,"applied":True,"state_index":len(states)-1,"record":plain(event)})
    if len(states)-1!=expected:
        raise ValueError("Saved accepted_steps differs from complete replayed trace")
    if graph_signature(current)!=graph_signature(after):
        raise ValueError("Replayed final indexed graph/attributes do not equal the saved endpoint")
    return states,events


def trajectory_command(args):
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    before,after,bmeta,ameta,sources=load_pair(args.before_branch,args.after_branch)
    diagnostics,diagnostic_source=load_diagnostics(args.after_branch.resolve(),ameta,args.diagnostics)
    if diagnostic_source is None:
        raise ValueError("No hash-anchored trace artifact; rerun/export with accepted edge-swap traces enabled")
    ids=args.sample_ids or bmeta["requested_sample_ids"][:args.num_samples]
    if not ids or len(ids)!=len(set(ids)) or any(i not in bmeta["requested_sample_ids"] for i in ids):
        raise ValueError("Select distinct IDs from the matched requested population")
    replayed=[]
    for sample_id in ids:
        if sample_id not in before or sample_id not in after:
            raise ValueError(f"Requested ID {sample_id} has no saved endpoint pair; no trajectory can be reconstructed")
        if sample_id not in diagnostics:
            raise ValueError(f"No saved trace diagnostic for requested ID {sample_id}")
        states,events=replay_trace(before[sample_id],after[sample_id],diagnostics[sample_id],kind=args.kind,edge_attribute=args.edge_attribute)
        replayed.append((sample_id,states,events))
    columns=max(len(states) for _,states,_ in replayed)
    fig,axes=plt.subplots(len(ids),columns,figsize=(2.5*columns,2.8*len(ids)),squeeze=False,layout="constrained")
    manifest_rows=[]
    for row,(sample_id,states,events) in enumerate(replayed):
        union=nx.Graph()
        for graph in states:
            union.add_nodes_from(graph)
            union.add_edges_from(graph.edges())
        positions=nx.spring_layout(union,seed=args.layout_seed)
        panels=[]
        for column in range(columns):
            ax=axes[row,column]; ax.axis("off")
            if column>=len(states):
                continue
            graph=states[column]
            labels={node:str(graph.nodes[node].get(args.node_attribute,node)) for node in graph}
            nx.draw_networkx(graph,pos=positions,ax=ax,with_labels=args.node_labels,
                             labels=labels,node_size=85 if args.node_labels else 40,font_size=6,
                             width=.9,node_color="#297ba6",edge_color="#667580")
            connected=len(graph)>0 and nx.is_connected(graph)
            ax.set(xlim=(-1.15,1.15),ylim=(-1.15,1.15))
            ax.set_title(f"ID {sample_id} · {'source' if column==0 else 'swap '+str(column)}\nn={len(graph)}, m={graph.number_of_edges()}, connected={connected}",fontsize=8)
            panels.append({"state_index":column,"n":len(graph),"m":graph.number_of_edges(),"connected":connected,
                           "indexed_edges":[list(edge) for edge in graph.edges()]})
        manifest_rows.append({"sample_id":sample_id,"accepted_swaps":len(states)-1,"panels":panels,"events":events,
                              "coordinates":[{"node":plain(node),"x":float(p[0]),"y":float(p[1])} for node,p in positions.items()],
                              "rewiring":trace_statistics(diagnostics[sample_id]),"endpoint_verified":True})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_dir/"rewiring_trajectories.pdf",bbox_inches="tight")
    fig.savefig(args.output_dir/"rewiring_trajectories.png",dpi=args.dpi,bbox_inches="tight")
    plt.close(fig)
    evidence.write_json(args.output_dir/"rewiring_trajectory_manifest.json",{
        "format":FORMAT,"kind":"verified_accepted_swap_trajectories","source_files":sources,
        "diagnostic_source":diagnostic_source,"shared_context":bmeta["shared_context"],"graph_kind":args.kind,
        "selection":"explicit sample IDs" if args.sample_ids else "first requested IDs, no failure replacement",
        "layout":"one fixed spring layout of all recorded states' indexed edge union per sample; same axes for every step",
        "layout_seed":args.layout_seed,"samples":manifest_rows,
        "rendering":"Each accepted swap yields one panel; recorded rejection/stop decisions remain in events without a duplicate graph panel",
        "replay_validation":"Every removal/addition applicable; indexed degrees preserved; final indexed graph and node/edge attributes equal saved endpoint",
        "provenance_scope":"branch/common-index consistency, graph hashes, requested/successful IDs and shared context; training checkpoint/configuration checks remain the separate check_grapher_ablation_artifacts.py gate",
        "molecular_edge_attributes":"Replay requires identical removed edge payloads because the original trace records endpoints, not full attribute-to-edge assignments"})
    print(f"Replayed and verified {len(ids)} saved trajectories; no unrecorded intermediate states.")
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name,handler in (("drift",drift_command),("trajectory",trajectory_command)):
        sub=commands.add_parser(name)
        sub.add_argument("--before-branch",type=Path,required=True)
        sub.add_argument("--after-branch",type=Path,required=True)
        sub.add_argument("--diagnostics",type=Path,help="Optional choice among hash-declared branch/parent diagnostic files")
        sub.add_argument("--node-attribute",default="atomic_num")
        sub.add_argument("--edge-attribute",default="bond_type")
        sub.add_argument("--output-dir",type=Path,required=True)
        sub.set_defaults(handler=handler)
    drift=commands.choices["drift"]
    drift.add_argument("--graphlet-backend",choices=("exact","sampled"),default="exact")
    drift.add_argument("--graphlet-samples",type=int)
    drift.add_argument("--graphlet-seed",type=int,default=0)
    drift.add_argument("--max-graphlet-order",type=int,choices=(3,4,5),default=5)
    drift.add_argument("--clustering-bins",type=int,default=100)
    drift.add_argument("--spectral-bins",type=int,default=200)
    drift.add_argument("--plot",action="store_true")
    trajectory=commands.choices["trajectory"]
    trajectory.add_argument("--kind",choices=("generic","molecular"),required=True)
    trajectory.add_argument("--sample-ids",nargs="+")
    trajectory.add_argument("--num-samples",type=int,default=3)
    trajectory.add_argument("--layout-seed",type=int,default=42)
    trajectory.add_argument("--node-labels",action="store_true")
    trajectory.add_argument("--dpi",type=int,default=180)
    args=parser.parse_args(argv)
    try:
        if args.command=="drift":
            if args.clustering_bins<1 or args.spectral_bins<1:
                raise ValueError("Histogram bins must be positive")
            if args.graphlet_backend=="sampled" and (args.graphlet_samples is None or args.graphlet_samples<1):
                raise ValueError("Sampled graphlets require an explicit positive --graphlet-samples")
            if args.graphlet_backend=="exact" and args.graphlet_samples is not None:
                raise ValueError("--graphlet-samples applies only to the sampled backend")
        elif args.num_samples<1 or args.dpi<1:
            raise ValueError("Plot counts and DPI must be positive")
        return args.handler(args)
    except (OSError,ValueError,TypeError,KeyError,ImportError) as error:
        parser.exit(2,f"{type(error).__name__}: {error}\n")


if __name__=="__main__":
    raise SystemExit(main())
