"""CPU-only evidence contract tests; no model fitting or downloads."""
import importlib.util
import ast
import json
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_grapher_paper_experiments.py"
spec = importlib.util.spec_from_file_location("paper_evidence", SCRIPT)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        import networkx as nx
        self.nx = nx
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def graph_file(self, path, graphs):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(graphs, stream)
        return path

    def fixture(self, seed=41, values=None):
        root = self.root / f"seed{seed}" / "ablations"
        ids = [f"{seed}:0", f"{seed}:1"]
        for variant, value in (values or {"T":1.,"D":3.,"TR":2.,"DR":5.}).items():
            branch = root / variant
            graphs = [self.nx.path_graph(4), self.nx.cycle_graph(4)]
            for sample_id, graph in zip(ids, graphs):
                graph.graph["sample_id"] = sample_id
            path = self.graph_file(branch / "base_graphs.pkl", graphs)
            metadata = {"status":"complete","sample_ids":ids,"requested_sample_ids":ids,
                        "requested_count":2,"completed_count":2,"failed_count":0,"failures":[],
                        "shared_context":{"parent_hash":f"fixture-{seed}"},
                        "base_graphs":{"path":path.name,"sha256":evidence.sha256(path)}}
            evidence.write_json(branch / "manifest.json",metadata)
            report = {"generated_stage":variant,"reference_split":"test",
                      "generated_graphs":str(path),"generated_graphs_sha256":evidence.sha256(path),
                      "reference_split_sha256":"frozen-test", "num_generated_graphs_evaluated":2,
                      "num_reference_graphs":3,
                      "metrics":[{"comparison":f"{variant}_to_test","degree_mmd":value,
                                  "joint_vun_percent":50}],
                      "generic_exact_isomorphism_diagnostics":{"protocol":{},"population_metrics":[
                          {"population":variant,"num_valid_graphs":2,"num_exact_isomorphism_classes":2,
                           "num_exact_novel_graphs_against_train":1,
                           "exact_isomorphism_novelty_rate_against_train":.5,
                           "exact_isomorphism_uniqueness_rate":1}]}}
            evidence.write_json(branch / "evaluation_test" / "graph_evaluation_report.json",report)
        return root

    def invoke(self, *args, expected=0):
        result = subprocess.run([sys.executable,str(SCRIPT),*map(str,args)],capture_output=True,text=True)
        self.assertEqual(result.returncode,expected,result.stderr+result.stdout)
        return result

    def test_cli_audit_unknown_attempts_and_timing_stay_null(self):
        root = self.fixture()
        output = self.root / "audit"
        self.invoke("audit","--seed-dir",f"41={root}","--output-dir",output)
        data = evidence.read_json(output / "paper_evidence_audit.json")
        for row in data["runs"]:
            self.assertIsNone(row["counts"]["N_att"])
            self.assertTrue(all(v is None for v in row["runtime"].values()))
            self.assertEqual(row["counts"]["N_out"],2)
            self.assertEqual(row["artifact_integrity"],"branch_hash_and_ids_verified")
        self.assertEqual(len(data["invariant_pairs"]),2)
        self.assertEqual(data["invariant_pairs"][0]["num_pairs"],2)

    def test_real_sample_sd_and_within_seed_interaction(self):
        one = self.fixture(41)
        two = self.fixture(42,{"T":2.,"D":4.,"TR":3.,"DR":8.})
        output = self.root / "summary"
        self.invoke("summarize","--seed-dir",f"41={one}","--seed-dir",f"42={two}","--output-dir",output)
        data = evidence.read_json(output / "paper_experiment_summary.json")
        row = next(r for r in data["contrast_summary"] if r["contrast"]=="decoder_rewiring_interaction" and r["metric"]=="degree_mmd")
        self.assertEqual(row["n"],2)
        self.assertEqual(row["mean"],2.)
        self.assertAlmostEqual(row["std"],math.sqrt(2))

    def test_mismatched_requested_ids_block_matched_contrasts(self):
        root = self.fixture()
        path = root / "D" / "manifest.json"
        data = evidence.read_json(path)
        data["requested_sample_ids"] = ["other:0","other:1"]
        evidence.write_json(path,data)
        output = self.root / "summary"
        self.invoke("summarize","--seed-dir",f"41={root}","--output-dir",output)
        data = evidence.read_json(output / "paper_experiment_summary.json")
        row = next(r for r in data["paired_differences"] if r["contrast"]=="D_minus_T" and r["metric"]=="degree_mmd")
        self.assertIsNone(row["difference"])
        self.assertEqual(row["status"],"unverified_or_mismatched_requested_ids")

    def test_hash_changes_and_embedded_id_mismatch_are_rejected(self):
        root = self.fixture()
        path = root / "T" / "base_graphs.pkl"
        graphs = evidence.load_graphs(path)
        graphs[0].graph["sample_id"] = "bad"
        self.graph_file(path,graphs)
        with self.assertRaisesRegex(ValueError,"hash-mismatched"):
            evidence.checked_population(root / "T")
        metadata = evidence.read_json(root / "T" / "manifest.json")
        metadata["base_graphs"]["sha256"] = evidence.sha256(path)
        evidence.write_json(root / "T" / "manifest.json",metadata)
        with self.assertRaisesRegex(ValueError,"Embedded graph ID"):
            evidence.checked_population(root / "T")

    def test_missing_diagnostics_do_not_count_as_false(self):
        result = evidence.aggregate_diagnostic_rows([{"degree_exact":True},{"degree_exact":None},{}])
        self.assertEqual(result["degree_exact"],{"n_measured":1,"n_missing":2,"true_count":1,"rate":1.})

    def test_coverage_is_permutation_invariant_and_exact_epsilon_rule(self):
        a = self.nx.path_graph(5)
        b = self.nx.relabel_nodes(a,{i:10-i for i in a})
        self.assertEqual(evidence.structural_distance(evidence.structural_profile(a,3),evidence.structural_profile(b,3)),0)
        data = evidence.compute_coverage({"train":[a],"test":[b],"T":[b]},"train","test",[0.,.1],3)
        row = next(r for r in data["epsilon_curve"] if r["population"]=="T" and r["epsilon"]==0.)
        self.assertEqual(row["epsilon_novelty_rate_against_train"],0.)
        self.assertEqual(row["heldout_coverage_rate"],1.)

    def test_coverage_cli_records_hashes_and_creates_plot(self):
        paths = [self.graph_file(self.root / f"{name}.pkl",[self.nx.path_graph(4)]) for name in ("train","test","T")]
        output = self.root / "coverage"
        self.invoke("coverage","--train-graphs",paths[0],"--heldout-graphs",paths[1],
                    "--reference-split","test","--population",f"T={paths[2]}",
                    "--epsilons",0,.1,"--wl-iterations",3,"--plot","--output-dir",output)
        data = evidence.read_json(output / "graded_coverage_report.json")
        self.assertEqual(data["source_files"]["T"]["sha256"],evidence.sha256(paths[2]))
        self.assertEqual(data["protocol"]["epsilons"],[0.,.1])
        self.assertTrue((output / "graded_coverage.pdf").is_file())

    def test_distance_matches_the_existing_evaluator(self):
        # Execute only the dependency-light distance functions from the existing
        # evaluator, avoiding a torch/ORCA/MMD import merely to test distance.
        import hashlib
        from collections import Counter
        from typing import Any, Mapping, Sequence
        names = {"_stable_wl_digest", "_normalized_counter", "_tv_histogram",
                 "_component_size_histogram", "_wl_layer_histograms",
                 "_graded_topology_profile", "graded_topology_distance"}
        tree = ast.parse((ROOT / "scripts" / "evaluate_graph_generation_report.py").read_text())
        selected = ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
        ns = {"hashlib":hashlib,"Counter":Counter,"Any":Any,"Mapping":Mapping,"Sequence":Sequence,"nx":self.nx}
        exec(compile(selected,"existing_evaluator_distance","exec"),ns)
        graphs = [self.nx.path_graph(6),self.nx.cycle_graph(6),self.nx.complete_graph(5),self.nx.empty_graph(3),self.nx.Graph()]
        for left in graphs:
            for right in graphs:
                actual = evidence.structural_distance(evidence.structural_profile(left,3),evidence.structural_profile(right,3))
                self.assertAlmostEqual(actual,ns["graded_topology_distance"](left,right,wl_iterations=3))

    def test_independent_training_seed_contrast_uses_explicit_paths(self):
        root = self.fixture()
        output = self.root / "training_summary"
        self.invoke("summarize","--variants","T","S1","--run",f"41,T={root / 'T'}",
                    "--run",f"41,S1={root / 'D'}","--pairing","seed",
                    "--contrast","S1_minus_T=S1,T","--output-dir",output)
        data = evidence.read_json(output / "paper_experiment_summary.json")
        row = next(r for r in data["contrast_summary"] if r["contrast"]=="S1_minus_T" and r["metric"]=="degree_mmd")
        self.assertEqual(row["mean"],2.)
        self.assertIsNone(row["std"])

    def test_matched_gallery_records_shared_coordinates_and_failures(self):
        root = self.fixture()
        branch = root / "DR"
        graphs = evidence.load_graphs(branch / "base_graphs.pkl")
        self.graph_file(branch / "base_graphs.pkl",graphs[:1])
        data = evidence.read_json(branch / "manifest.json")
        data.update(status="partial",sample_ids=["41:0"],completed_count=1,failed_count=1,
                    failures=[{"sample_id":"41:1","error":"fixture failure"}])
        data["base_graphs"]["sha256"] = evidence.sha256(branch / "base_graphs.pkl")
        evidence.write_json(branch / "manifest.json",data)
        output = self.root / "gallery"
        self.invoke("plot","--ablation-dir",root,"--num-samples",2,"--output-dir",output)
        data = evidence.read_json(output / "matched_gallery_manifest.json")
        self.assertEqual([r["sample_id"] for r in data["samples"]],["41:0","41:1"])
        self.assertEqual(data["samples"][1]["variants"]["DR"]["status"],"failed")
        self.assertEqual(len(data["samples"][0]["coordinates"]),4)
        self.assertTrue((output / "matched_T_D_TR_DR.png").is_file())

    def test_empty_run_is_reported_without_fabricating_measurements(self):
        output = self.root / "missing"
        self.invoke("audit","--seed-dir",f"41={self.root / 'absent'}","--metadata-only","--output-dir",output)
        data = evidence.read_json(output / "paper_evidence_audit.json")
        self.assertTrue(all(r["counts"]["N_att"] is None for r in data["runs"]))
        self.assertTrue(all(r["counts"]["N_out"] is None for r in data["runs"]))

    def test_direct_baseline_population_counts_do_not_invent_sample_ids(self):
        root = self.fixture()
        branch = root / "T"
        path = branch / "manifest.json"
        metadata = evidence.read_json(path)
        evidence.write_json(path,{"model_id":"fixture_baseline","num_requested":2,"num_generated":2,
                                  "base_graphs":metadata["base_graphs"],"duration_seconds":1.2})
        output = self.root / "baseline_audit"
        self.invoke("audit","--run",f"41,baseline={branch}","--variants","baseline","--output-dir",output)
        row = evidence.read_json(output / "paper_evidence_audit.json")["runs"][0]
        self.assertEqual(row["counts"]["N_out"],2)
        self.assertEqual(row["counts"]["N_req"],2)
        self.assertIsNone(row["counts"]["N_att"])
        self.assertIsNone(row["sample_ids"])
        self.assertEqual(row["artifact_integrity"],"population_hash_verified_unmatched")


if __name__ == "__main__":
    unittest.main()
