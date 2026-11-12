"""Tiny CPU forward/measurement tests; synthetic checkpoints are not results."""
from __future__ import annotations

import importlib.util
import json
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostics = load_module("prediction_diagnostics",ROOT / "scripts" / "evaluate_grapher_prediction_diagnostics.py")
profile = load_module("profile_paper",ROOT / "scripts" / "profile_grapher_paper.py")


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def setUp(self):
        import networkx as nx
        self.nx=nx
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def pickle(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(value,stream)
        return {"path":path.name,"sha256":profile.sha256(path)}

    def heldout_fixture(self,molecular=False):
        import torch
        from grapher.models.grapher import vanilla_gsdm as v
        graphs=[self.nx.path_graph(4),self.nx.cycle_graph(4)]
        cfg={"enabled":True,"graphlet":{"orders":[3]},"clustering":{"enabled":False},
             "orbit":{"enabled":False},"degree":{"enabled":False}}
        _targets,meta=v._joint_structure_targets(graphs,cfg)
        mc={"max_nodes":4,"max_feat_num":4,"hidden_dim":8,"depth":2,
            "ppgn_hidden_dim":8,"ppgn_depth":1,"structural_features":{"enabled":False}}
        options={"structure_summary":cfg,"sample":{"use_ema":False}}
        if molecular:
            from grapher.models.grapher import hybrid_attributed_loggap as h
            mc.update(node_classes=4,edge_classes=3)
            attribute_meta={"graphlet_slices":[[0,2]],"typed_degree_enabled":False}
            mx,ml=h._build_models(mc,meta,attribute_meta,torch.device("cpu"))
            extra={"format":h.CHECKPOINT_FORMAT,"topology_summary":meta,"attribute_summary":attribute_meta}
            options["topology_summary"]=cfg
        else:
            mx,ml=v._build_models(mc,torch.device("cpu"),graphlet_slices=tuple(tuple(p) for p in meta["graphlet_slices"]),summary_meta=meta)
            extra={"structure_summary":meta}
        state={"model_config":mc,"model_x_state":mx.state_dict(),"model_spectrum_state":ml.state_dict(),
               "spectral_transform":{"kind":"laplacian_log_gap","mean":[0.]*4,"std":[1.]*4,"epsilon":1e-6,"exp_clip":10.},
               "sde":{"x":{"beta_min":.1,"beta_max":1.,"num_scales":10},"spectrum":{"beta_min":.1,"beta_max":1.,"num_scales":10},"eigen_mask":"laplacian_nonzero_prefix"},**extra}
        checkpoint=self.root / "run" / "train" / "checkpoint.pt"
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(state,checkpoint)
        self.pickle(self.root / "datasets" / "tiny" / "test.pkl",graphs)
        args=SimpleNamespace(run_dir=self.root/"run",device="cpu",dataset="qm9" if molecular else "community_small",
                             dataset_root=self.root/"datasets",serialized_dataset="tiny",split="test",max_graphs=None,
                             seed=41,times=[.2,.6],noise_repeats=2,batch_size=1)
        training={"checkpoint":{"path":checkpoint.name,"sha256":profile.sha256(checkpoint)},"options":options}
        return args,training

    def test_generic_heldout_forward_is_reproducible_and_separate_from_generation(self):
        args,training=self.heldout_fixture()
        one=diagnostics.heldout_diagnostics(args,training,{"checkpoint":"fixture"})
        two=diagnostics.heldout_diagnostics(args,training,{"checkpoint":"fixture"})
        self.assertEqual(one,two)
        self.assertEqual(len(one["per_graph"]),8)
        self.assertFalse(one["protocol"]["unconditional_generation_metric"])
        self.assertIn("standardized_log_gap_rmse",one["summary"])
        self.assertTrue(all(math.isfinite(row["spectrum_rmse"]) for row in one["per_graph"]))

    def test_molecular_heldout_forward_uses_real_hybrid_checkpoint_interface(self):
        args,training=self.heldout_fixture(molecular=True)
        result=diagnostics.heldout_diagnostics(args,training,{})
        self.assertEqual(len(result["per_graph"]),8)
        self.assertIn("graphlet_histogram_mae",result["summary"])
        self.assertTrue(math.isfinite(result["summary"]["standardized_log_gap_rmse"]))

    def test_heldout_rejects_modified_checkpoint(self):
        args,training=self.heldout_fixture()
        training["checkpoint"]["sha256"]="bad"
        with self.assertRaisesRegex(ValueError,"hash mismatch"):
            diagnostics.heldout_diagnostics(args,training,{})

    def test_endpoint_spectrum_and_missing_degree_diagnostics(self):
        import numpy as np
        graphs=[self.nx.path_graph(4),self.nx.cycle_graph(4)]
        ids=["41:0","41:1"]
        generation=self.root/"generation"
        (generation/"ablations").mkdir(parents=True)
        spectra=[]
        for sample_id,graph in zip(ids,graphs):
            graph.graph["sample_id"]=sample_id
            a=self.nx.to_numpy_array(graph)
            spectra.append(np.linalg.eigvalsh(np.diag(a.sum(axis=1))-a))
        manifest={"sampled_spectra":self.pickle(generation/"sampled_spectra.pkl",spectra)}
        diagnostics.write(generation/"ablations"/"ablation_manifest.json",{"sample_ids":ids})
        for variant in ("T","D"):
            branch=generation/"ablations"/variant
            record=self.pickle(branch/"base_graphs.pkl",graphs)
            diagnostics.write(branch/"manifest.json",{"sample_ids":ids,"base_graphs":record,"failures":[]})
        args=SimpleNamespace(generation_dir=generation,seed=41,num_samples=2,dataset="community_small")
        result=diagnostics.endpoint_diagnostics(args,manifest,{})
        self.assertAlmostEqual(result["summary"]["T"]["realized_spectrum_rmse_mean"],0.)
        self.assertEqual(result["summary"]["D"]["exact_degree_denominator"],0)
        self.assertIsNone(result["summary"]["D"]["exact_degree_per_completed"])

    def test_diagnostic_rows_reject_cross_sample_and_duplicate_id_pairing(self):
        with self.assertRaisesRegex(ValueError,"disagrees"):
            diagnostics.diagnostic_index([{"sample_id":"41:1","graph_index":0}],["41:0","41:1"])
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            diagnostics.diagnostic_index([{"sample_id":"41:0"},{"sample_id":"41:0"}],["41:0","41:1"])

    def test_nested_recorder_exclusive_time_and_failure_counts(self):
        from grapher.models.grapher import paper_measurement as pm
        rec=pm.Recorder()
        with mock.patch.object(pm,"perf_counter",side_effect=[0.,1.,3.,5.]):
            with rec.activate(),pm.stage("outer"):
                pm.attempted(3)
                with pm.stage("inner"):
                    pm.completed(2)
        self.assertEqual(rec.stages["outer"]["inclusive_seconds"],5.)
        self.assertEqual(rec.stages["outer"]["exclusive_seconds"],3.)
        self.assertEqual(rec.stages["inner"]["exclusive_seconds"],2.)
        self.assertEqual(rec.attempted_samples,3)
        self.assertEqual(rec.completed_endpoints,2)
        with self.assertRaises(ValueError):
            with rec.activate(),pm.stage("failure"):
                raise ValueError("fixture")
        self.assertEqual(rec.stages["failure"]["failures"],1)
        self.assertEqual(rec.stack,[])

    def run_profile(self,fail=False):
        from grapher.models.grapher import vanilla_gsdm as v
        original=v._padded_dataset
        target=self.root/("failed_target.py" if fail else "target.py")
        target.write_text("from grapher.models.grapher import paper_measurement as pm\n"
                          "import networkx as nx\n"
                          "from grapher.models.grapher import vanilla_gsdm as v\n"
                          "with pm.stage('test_outer'):\n"
                          "    pm.attempted(2)\n"
                          "    v._padded_dataset([nx.path_graph(4)],max_nodes=4,max_feat_num=4,spectral_operator='combinatorial_laplacian')\n"
                          "    pm.completed(1)\n"+
                          ("    raise RuntimeError('synthetic fixture failure')\n" if fail else ""))
        output=self.root/("failed_profile.json" if fail else "profile.json")
        args=["--output",str(output),"--label","fixture","--requested-count","2","--target-script",str(target),"--",
              "--stage","generate","--dataset","community_small","--run-id","fixture","--generation-id","fresh",
              "--num-samples","2","--device","cpu","--wrapper-config",str(self.root/"config.yaml"),
              "--output-root",str(self.root/"outputs")]
        # A synthetic target has no trained checkpoint; test the profiler's
        # real wrappers/bookkeeping with the independent verifier stubbed only.
        import check_grapher_ablation_artifacts as check
        with mock.patch.object(check,"training",return_value=({},{})),mock.patch.object(check,"config_options",return_value={}):
            code=profile.main(args)
        self.assertIs(v._padded_dataset,original)
        return code,json.loads(output.read_text())

    def test_profile_measures_work_and_restores_wrappers(self):
        code,result=self.run_profile()
        self.assertEqual(code,0)
        self.assertEqual(result["status"],"complete")
        self.assertEqual(result["attempted_samples"],2)
        self.assertEqual(result["completed_endpoints_before_rewiring"],1)
        self.assertGreater(result["stages"]["spectral_preprocessing"]["exclusive_seconds"],0.)
        self.assertIsNone(result["peak_gpu_allocated_bytes"])

    def test_profile_retains_failed_work(self):
        code,result=self.run_profile(fail=True)
        self.assertEqual(code,1)
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["attempted_samples"],2)
        self.assertEqual(result["stages"]["test_outer"]["failures"],1)
        self.assertGreater(result["wall_seconds"],0)


if __name__=="__main__":
    unittest.main()
