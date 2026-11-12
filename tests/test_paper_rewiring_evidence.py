"""Exact synthetic swap traces exercise replay, drift and artifact binding."""
import importlib.util
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/"scripts"/"analyze_grapher_rewiring_evidence.py"
spec=importlib.util.spec_from_file_location("rewiring_evidence",SCRIPT)
rewiring=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rewiring)


class RewiringEvidenceTests(unittest.TestCase):
    def setUp(self):
        import networkx as nx
        self.nx=nx
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def graphs(self,molecular=False):
        before=self.nx.Graph([(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)])
        if molecular:
            self.nx.set_node_attributes(before,6,"atomic_num")
            self.nx.set_edge_attributes(before,1,"bond_type")
        after=before.copy()
        after.remove_edges_from([(0,1),(3,4)])
        after.add_edges_from([(0,4),(1,3)],**({"bond_type":1} if molecular else {}))
        before.graph["sample_id"]="41:0"
        after.graph["sample_id"]="41:0"
        return before,after

    def diagnostic(self,molecular=False):
        action=[[[0,1],[3,4]],[[0,4],[1,3]]]
        if molecular:
            trace=[{"step":1,"removed":action[0],"added":action[1],
                    "candidate_search":{"proposals":4},"candidates_inside_trust_region":2}]
        else:
            trace=[{"step":0,"accepted_step":1,"accepted":True,"action":action,
                    "num_proposals":4,"num_eligible_candidates":2,"terminal_stop":False}]
        return {"sample_id":"41:0","graph_index":0,"requested_steps":1,"accepted_steps":1,"trace":trace}

    def fixture(self,molecular=False,parent_pickle=False):
        before,after=self.graphs(molecular)
        pair=[]
        for variant,graph in (("T",before),("TR",after)):
            branch=self.root/"generation"/"ablations"/variant
            branch.mkdir(parents=True)
            path=branch/"base_graphs.pkl"
            with path.open("wb") as stream:
                pickle.dump([graph],stream)
            meta={"requested_sample_ids":["41:0"],"sample_ids":["41:0"],"requested_count":1,
                  "completed_count":1,"failed_count":0,"failures":[],"status":"complete",
                  "shared_context":{"parent":"fixture"},
                  "base_graphs":{"path":path.name,"sha256":rewiring.evidence.sha256(path)}}
            if variant=="TR" and not parent_pickle:
                path=branch/"diagnostics.json"
                rewiring.evidence.write_json(path,{"per_graph":[self.diagnostic(molecular)]})
                meta["diagnostics"]={"path":path.name,"sha256":rewiring.evidence.sha256(path)}
            rewiring.evidence.write_json(branch/"manifest.json",meta)
            pair.append(branch)
        if parent_pickle:
            path=self.root/"generation"/"graphlet_refinement_diagnostics.pkl"
            with path.open("wb") as stream:
                pickle.dump([self.diagnostic(molecular)],stream)
            rewiring.evidence.write_json(path.parent/"manifest.json",{
                "graphlet_refinement_diagnostics":{"path":path.name,"sha256":rewiring.evidence.sha256(path)}})
        return pair

    def invoke(self,*args,expected=0):
        result=subprocess.run([sys.executable,str(SCRIPT),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
        return result

    def test_generic_replay_matches_endpoint_and_fails_on_incomplete_trace(self):
        before,after=self.graphs()
        states,events=rewiring.replay_trace(before,after,self.diagnostic(),kind="generic",edge_attribute="bond_type")
        self.assertEqual(len(states),2)
        self.assertEqual(rewiring.graph_signature(states[-1]),rewiring.graph_signature(after))
        bad=self.diagnostic(); bad["trace"]=[]
        with self.assertRaisesRegex(ValueError,"accepted_steps"):
            rewiring.replay_trace(before,after,bad,kind="generic",edge_attribute="bond_type")
        with self.assertRaisesRegex(ValueError,"Missing saved trace"):
            rewiring.replay_trace(before,before,{"accepted_steps":0},kind="generic",edge_attribute="bond_type")
        with self.assertRaisesRegex(ValueError,"saved endpoint"):
            rewiring.replay_trace(before,before,self.diagnostic(),kind="generic",edge_attribute="bond_type")

    def test_molecular_payloads_are_preserved_and_ambiguity_rejected(self):
        before,after=self.graphs(molecular=True)
        states,_events=rewiring.replay_trace(before,after,self.diagnostic(True),kind="molecular",edge_attribute="bond_type")
        self.assertEqual(states[-1].edges[0,4]["bond_type"],1)
        before.edges[0,1]["unrecorded_payload"]="a"
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            rewiring.replay_trace(before,after,self.diagnostic(True),kind="molecular",edge_attribute="bond_type")

    def test_trajectory_cli_reads_parent_pickle_and_exports_shared_layout(self):
        before,after=self.fixture(parent_pickle=True)
        output=self.root/"trajectory"
        self.invoke("trajectory","--before-branch",before,"--after-branch",after,"--kind","generic",
                    "--sample-ids","41:0","--output-dir",output)
        result=rewiring.evidence.read_json(output/"rewiring_trajectory_manifest.json")
        self.assertEqual(result["samples"][0]["accepted_swaps"],1)
        self.assertEqual(len(result["samples"][0]["coordinates"]),6)
        self.assertTrue(result["samples"][0]["endpoint_verified"])
        self.assertTrue((output/"rewiring_trajectories.pdf").is_file())

    def test_drift_cli_detects_nonzero_structure_change(self):
        before,after=self.fixture()
        output=self.root/"drift"
        self.invoke("drift","--before-branch",before,"--after-branch",after,"--max-graphlet-order",4,
                    "--plot","--output-dir",output)
        result=rewiring.evidence.read_json(output/"rewiring_drift_report.json")
        row=result["per_sample"][0]
        self.assertGreater(row["metrics"]["clustering_histogram_tv"],0)
        self.assertGreater(row["metrics"]["orbit_log1p_count_rmse"],0)
        self.assertGreater(row["metrics"]["graphlet_k3_histogram_tv"],0)
        self.assertEqual(row["rewiring"]["proposals"],4)
        self.assertEqual(result["invariants"]["fields"]["indexed_degrees_preserved"]["rate"],1.)
        self.assertTrue((output/"rewiring_drift_ecdf.pdf").is_file())

    def test_sampled_graphlets_share_subsets_and_keep_protocol_explicit(self):
        before,after=self.fixture()
        output=self.root/"sampled"
        self.invoke("drift","--before-branch",before,"--after-branch",after,"--graphlet-backend","sampled",
                    "--graphlet-samples",10,"--graphlet-seed",71,"--output-dir",output)
        result=rewiring.evidence.read_json(output/"rewiring_drift_report.json")
        self.assertEqual(result["protocol"]["graphlet_samples_per_order"],10)
        self.assertEqual(result["protocol"]["graphlet_seed"],71)
        from grapher.rewiring_mlp.generic.basis import TopologyGraphletBasis
        from grapher.rewiring_mlp.properties.summary import SummaryConfig
        cfg=SummaryConfig.from_dict({"graphlet_k_min":3,"graphlet_k_max":5,"graphlet_connected_only":True})
        basis=TopologyGraphletBasis.from_config(cfg)
        g,_=self.graphs()
        h=self.nx.Graph()
        h.add_nodes_from(reversed(list(g.nodes())))
        h.add_edges_from(reversed(list(g.edges())))
        args=SimpleNamespace(graphlet_backend="sampled",graphlet_samples=10,graphlet_seed=71,
                             spectral_bins=200,clustering_bins=100)
        values,_=rewiring.paired_drift(g,h,basis,cfg,args,"41:0")
        self.assertTrue(all(abs(value)<1e-9 for value in values.values()))

    def test_diagnostic_hash_mismatch_and_partial_proposals_not_silently_used(self):
        _before,after=self.fixture()
        with (after/"diagnostics.json").open("a") as stream:
            stream.write(" ")
        metadata=rewiring.evidence.read_json(after/"manifest.json")
        with self.assertRaisesRegex(ValueError,"hash-mismatched"):
            rewiring.load_diagnostics(after,metadata)
        row={"requested_steps":4,"accepted_steps":1,"trace":[{"num_proposals":8},{}]}
        result=rewiring.trace_statistics(row)
        self.assertIsNone(result["proposals"])
        self.assertEqual(result["recorded_proposal_sum"],8)
        self.assertEqual(result["proposal_decisions_measured"],1)
        self.assertIsNone(result["eligible_candidates_recorded"])


if __name__=="__main__":
    unittest.main()
