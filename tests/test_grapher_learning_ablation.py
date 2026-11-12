"""Behavioral tests for paper learning controls; no datasets or GPU required."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from grapher.experiments.learning_ablation import (
    DATASETS, MOLECULAR, applicable_controls, build_control_config,
    initialize_frozen_topology, assert_frozen_topology,
    require_attribute_reference, sha256, verify_attribute_population,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def config(dataset="community_small", seed=42):
    import yaml
    suffix = "paired" if dataset in MOLECULAR else "matched"
    return yaml.safe_load((REPO / f"configs/experiments/grapher_paper_ablation/{dataset}_seed_{seed}_{suffix}.yaml").read_text())


class PlanTests(unittest.TestCase):
    def test_missing_or_inactive_interventions_fail(self):
        with self.assertRaisesRegex(ValueError, "N/A"):
            build_control_config(config("qm9"), "qm9", "S3", 42)
        base = config()
        base["grapher"]["structure_summary"]["orbit"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "no active"):
            build_control_config(base, "community_small", "S2", 42)
        base = config("zinc")
        base["grapher"]["attribute_summary"]["loss_weight"] = 0.0
        with self.assertRaisesRegex(ValueError, "no-op"):
            build_control_config(base, "zinc", "A2", 42)

    def test_removal_preserves_capacity_and_t_budget(self):
        for dataset in DATASETS:
            for variant in applicable_controls(dataset):
                base = config(dataset)
                change = build_control_config(base, dataset, variant, 42)["grapher"]
                self.assertEqual(change["train"]["epochs"], 200)
                self.assertEqual(change["model"], base["grapher"]["model"])
                self.assertFalse(change["sample"]["degree_constrained_realization"]["enabled"])
                self.assertFalse(change["sample"]["matched_ablation"]["enabled"])
                summary = "topology_summary" if dataset in MOLECULAR else "structure_summary"
                for task in ("graphlet", "degree", "clustering", "orbit"):
                    if task in base["grapher"][summary]:
                        self.assertEqual(change[summary][task]["enabled"], base["grapher"][summary][task]["enabled"])

    def test_help_and_plan_without_neural_packages(self):
        code = """
import importlib.abc, sys, runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch', 'networkx', 'scipy'}:
   raise RuntimeError('neural dependency imported: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['scripts/run_grapher_learning_ablation.py', '--dataset', 'qm9', '--variants', 'A1', 'A2', '--seeds', '42', '--stage', 'plan']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Adding required A0", result.stdout)
        self.assertIn("REQUIRES", result.stdout)
        self.assertIn("categorical logits have no direct spectral input", result.stdout)


    def test_runner_publishes_then_reuses_verified_population(self):
        import contextlib
        import io
        import yaml
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("learning_runner_roundtrip_test", REPO / "scripts/run_grapher_learning_ablation.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs"
            run = output / "grapher/community_small/seed_42_community_small_paper_learning_v1_S4"
            generation = run / "generations/paper_learning_v1_seed_42_n_2"
            evaluation = generation / "evaluation_test"
            executed = []
            def fake_run(command, seed, args):
                executed.append(command)
                if "--stage" in command:
                    stage = command[command.index("--stage") + 1]
                    if stage == "train":
                        checkpoint = run / "train/checkpoints/grapher.pt"
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint.write_bytes(b"trained fixture")
                        options = yaml.safe_load(Path(command[command.index("--wrapper-config") + 1]).read_text())["grapher"]
                        (run / "train/manifest.json").write_text(json.dumps({"checkpoint": {"sha256": sha256(checkpoint)}, "checkpoint_selection": {"epoch": 200}, "train_seed": 42, "options": options}))
                    else:
                        self.assertFalse(generation.exists(), "Evaluation markers must not create a generation destination before atomic generation")
                        generation.mkdir(parents=True)
                        graphs = generation / "base_graphs.pkl"
                        graphs.write_bytes(b"population fixture")
                        (generation / "manifest.json").write_text(json.dumps({"checkpoint": {"sha256": sha256(run / "train/checkpoints/grapher.pt")}, "generation_seed": 42, "num_generated": 2, "num_requested": 2, "base_graphs": {"path": graphs.name, "sha256": sha256(graphs)}}))
                else:
                    self.assertEqual(json.loads((evaluation / "evaluation_status.json").read_text())["status"], "running")
            argv = ["--dataset", "community_small", "--variants", "S4", "--seeds", "42", "--stage", "all", "--num-samples", "2", "--output-root", str(output), "--report-root", str(root / "reports")]
            with patch.object(runner, "_run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(argv), 0)
                self.assertEqual(len(executed), 3)
                self.assertTrue((generation / "learning_generation_seal.json").is_file())
                self.assertEqual(json.loads((evaluation / "evaluation_status.json").read_text())["status"], "complete")
                executed.clear()
                argv[argv.index("all")] = "generate"
                self.assertEqual(runner.main(argv), 0)
                self.assertEqual(executed, [])

    def test_failed_reevaluation_marks_stale_reports_failed(self):
        import contextlib
        import io
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("learning_runner_status_test", REPO / "scripts/run_grapher_learning_ablation.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evaluation = output / "grapher/community_small/seed_42_community_small_paper_learning_v1_S4/generations/paper_learning_v1_seed_42_n_1024/evaluation_test"
            evaluation.mkdir(parents=True)
            status = evaluation / "evaluation_status.json"
            status.write_text(json.dumps({"status": "complete", "exit_code": 0}))
            (evaluation / "old_metrics.json").write_text('{"previous": true}')
            with patch.object(runner, "_one", side_effect=subprocess.CalledProcessError(17, ["evaluator"])), contextlib.redirect_stderr(io.StringIO()):
                result = runner.main(["--dataset", "community_small", "--variants", "S4", "--seeds", "42", "--stage", "evaluate", "--output-root", str(output)])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(status.read_text())["status"], "failed")
            self.assertEqual(json.loads(status.read_text())["exit_code"], 17)
            self.assertTrue((evaluation / "old_metrics.json").exists())


@unittest.skipUnless(HAS_TORCH, "torch is needed for actual gradient/checkpoint tests")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def inputs(self):
        import torch
        torch.manual_seed(123)
        x = torch.rand(2, 4, 4)
        adj = torch.tensor([[[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]] * 2, dtype=torch.float)
        flags = torch.ones(2, 4)
        u = torch.eye(4).repeat(2, 1, 1)
        state = torch.randn(2, 4)
        return x, adj, flags, u, state

    def generic_model(self, detached=False, dense=False):
        from grapher.models.grapher.vanilla_gsdm import GSDMSpectrumGraphletPPGNScore, GSDMSpectrumGraphletScore
        common = dict(max_feat_num=4, max_nodes=4, hidden_dim=4, depth=1,
                      graphlet_slices=((0, 2),), orbit_width=15, clustering_bins=3, degree_bins=4,
                      structural_features={"enabled": False}, detach_summary_encoder=detached)
        if dense:
            return GSDMSpectrumGraphletScore(**common)
        return GSDMSpectrumGraphletPPGNScore(**common, ppgn_hidden_dim=4, ppgn_depth=1)

    def molecular_model(self, detached=False, context="summary_log_gap"):
        from grapher.models.grapher.hybrid_attributed_loggap import HybridAttributedSpectrumPPGNScore
        return HybridAttributedSpectrumPPGNScore(max_feat_num=4, max_nodes=4, node_classes=2, edge_classes=2,
            topology_graphlet_slices=((0, 2),), typed_graphlet_slices=((0, 3),), structural_features={"enabled": False},
            topology_clustering_bins=3, topology_orbit_width=15, topology_degree_bins=4,
            typed_degree_bins=3, ppgn_hidden_dim=4, ppgn_depth=1,
            detach_topology_summary_encoder=detached, attribute_spectral_context=context)

    def test_resolved_configs_reach_actual_active_wrapper(self):
        import yaml
        from grapher.models.grapher.wrapper import GraphERWrapper
        for dataset in DATASETS:
            for seed in range(41, 46):
                for variant in applicable_controls(dataset):
                    item = build_control_config(config(dataset, seed), dataset, variant, seed)
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / "config.yaml"
                        path.write_text(yaml.safe_dump(item))
                        request = types.SimpleNamespace(config_path=path, options={}, run=types.SimpleNamespace(dataset_id=dataset))
                        resolved = GraphERWrapper()._options(request)
                    summary = "topology_summary" if dataset in MOLECULAR else "structure_summary"
                    if variant == "S4":
                        self.assertTrue(resolved[summary]["detach_encoder"])
                    if variant == "A2":
                        self.assertEqual(resolved["attributed"]["spectral_context"], "none")

    def test_s4_generic_heads_learn_encoder_detaches_and_denoiser_still_learns(self):
        import torch
        for dense in (False, True):
            baseline = self.generic_model(dense=dense)
            detached = self.generic_model(detached=True, dense=dense)
            detached.load_state_dict(baseline.state_dict())
            inputs = self.inputs()
            normal_out = baseline.forward_all(*inputs)
            stopped_out = detached.forward_all(*inputs)
            for key in normal_out:
                self.assertTrue(torch.equal(normal_out[key], stopped_out[key]), key)
            sum(value.square().mean() for key, value in stopped_out.items() if key != "spectrum").backward()
            encoder = detached.layers if dense else detached.pair_input
            self.assertTrue(all(value.grad is None for value in encoder.parameters()))
            self.assertGreater(sum(float(value.grad.abs().sum()) for value in detached.graphlet_logits.parameters()), 0)
            detached.zero_grad(set_to_none=True)
            detached.forward_all(*inputs)["spectrum"].square().mean().backward()
            self.assertGreater(sum(float(value.grad.abs().sum()) for value in encoder.parameters() if value.grad is not None), 0)

    def test_s4_molecular_topology_encoder_detaches_only_summary(self):
        model = self.molecular_model(detached=True)
        output = model.forward_topology(*self.inputs())
        sum(value.square().mean() for key, value in output.items() if key != "spectrum").backward()
        self.assertTrue(all(value.grad is None for value in model.topology_pair_input.parameters()))
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in model.topology_graphlet_logits.parameters()), 0)
        model.zero_grad(set_to_none=True)
        model.forward_topology(*self.inputs())["spectrum"].square().mean().backward()
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in model.topology_pair_input.parameters()), 0)

    def test_removed_tasks_have_zero_gradient(self):
        import torch
        from grapher.models.grapher.vanilla_gsdm import _structure_auxiliary_loss
        for variant, head_names in (("S1", ("graphlet_logits", "graphlet_mass_logits")), ("S2", ("orbit_histogram_logits", "orbit_log_total_raw")), ("S3", ("degree_histogram_logits",))):
            model = self.generic_model()
            out = model.forward_all(*self.inputs())
            summary = build_control_config(config(), "community_small", variant, 42)["grapher"]["structure_summary"]
            loss, _ = _structure_auxiliary_loss(model, out,
                torch.tensor([[.8,.2],[.2,.8]]), torch.tensor([[.3],[.7]]),
                torch.full((2,3), 1/3), torch.full((2,15), 1/15), torch.ones((2,1)), torch.full((2,4), 1/4),
                structure_cfg=summary)
            loss.backward()
            for name in head_names:
                self.assertTrue(all(value.grad is None or not bool(value.grad.count_nonzero()) for value in getattr(model, name).parameters()), (variant, name))
            self.assertGreater(sum(float(value.grad.abs().sum()) for value in model.pair_input.parameters() if value.grad is not None), 0)

    def test_a2_removes_exact_existing_spectral_side_channel(self):
        import torch
        model = self.molecular_model()
        ablated = self.molecular_model(context="none")
        ablated.load_state_dict(model.state_dict())
        inputs = self.inputs()
        x, adj, flags, u, state = inputs
        nodes, bonds = model.categorical_attribute_inputs(flags, adj.bool(), torch.zeros((2,4), dtype=torch.long), torch.zeros((2,4,4), dtype=torch.long))
        kw = dict(node_attr=nodes, edge_attr=bonds, attribute_time=torch.ones(2) * .5)
        ref1 = model.forward_attributes(*inputs, **kw)
        ref2 = model.forward_attributes(x, adj, flags, u, state + 3, **kw)
        changed1 = ablated.forward_attributes(*inputs, **kw)
        changed2 = ablated.forward_attributes(x, adj, flags, u, state + 3, **kw)
        self.assertFalse(torch.equal(ref1["typed_graphlet_logits"], ref2["typed_graphlet_logits"]))
        for key in changed1:
            self.assertTrue(torch.equal(changed1[key], changed2[key]), key)
        for key in ("node_logits", "edge_logits"):
            self.assertTrue(torch.equal(ref1[key], ref2[key]), "Existing categorical logits have no direct spectral channel")
        model.zero_grad(set_to_none=True)
        ablated.zero_grad(set_to_none=True)
        ref1["typed_graphlet_logits"].square().mean().backward()
        changed1["typed_graphlet_logits"].square().mean().backward()
        self.assertFalse(torch.equal(model.attribute_pair_input.weight.grad, ablated.attribute_pair_input.weight.grad), "A2 must change actual attribute training gradients")

    def test_attribute_optimizer_cannot_change_topology(self):
        import numpy as np
        import torch
        reference_x = torch.nn.Linear(4,4)
        reference_model = self.molecular_model()
        fresh_x = torch.nn.Linear(4,4)
        fresh_model = self.molecular_model(context="none")
        fresh_attribute = fresh_model.node_category_head.state_dict()
        cfg = {"max_nodes":4, "attribute_spectral_context":"none", "detach_topology_summary_encoder":False}
        meta = {"graphlet_slices":[[0,2]], "degree_bins":4, "orbit_width":15}
        transform = {"kind":"laplacian_log_gap", "mean":[0]*4, "std":[1]*4}
        basis = np.zeros((2,4,4), dtype=np.float32)
        sizes = np.array([4,4])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "grapher.pt"
            state = {"format":"grapher_laplacian_loggap_hybrid_attributed_ppgn_v1", "attribute_diffusion_training_contract":"fixed_topology_marginal_attribute_diffusion_v1",
                     "model_config":dict(cfg, attribute_spectral_context="summary_log_gap"), "topology_summary":meta,
                     "spectral_transform":transform, "basis_adjacencies":basis, "basis_num_nodes":sizes, "sde":{},
                     "model_x_state":reference_x.state_dict(), "model_spectrum_state":reference_model.state_dict(),
                     "ema_x_state":reference_x.state_dict(), "ema_spectrum_state":reference_model.state_dict()}
            torch.save(state,checkpoint)
            train_manifest = root / "train.json"
            train_manifest.write_text(json.dumps({"dataset":{"fingerprint":"dataset"}, "train_seed":42, "checkpoint_selection":{"epoch":200}}))
            for name in ("manifest.json", "topology_pairing.json", "sampled_basis_indices.pkl"):
                (root / name).write_text("fixture")
            ref = {"checkpoint":str(checkpoint), "checkpoint_sha256":sha256(checkpoint), "training_manifest":str(train_manifest), "training_manifest_sha256":sha256(train_manifest), "generation_dir":str(root), "generation_manifest_sha256":sha256(root/"manifest.json"), "topology_pairing_sha256":sha256(root/"topology_pairing.json"), "sampled_basis_indices_sha256":sha256(root/"sampled_basis_indices.pkl")}
            options = {"train":{"epochs":200,"attribute_only":True,"frozen_topology_reference":ref},"sde":{}}
            frozen = initialize_frozen_topology(fresh_x,fresh_model, options=options, dataset_fingerprint="dataset",train_seed=42, model_config=cfg,topology_meta=meta,spectral_transform=transform,basis_adjacencies=basis,basis_num_nodes=sizes)
            for key, value in fresh_attribute.items():
                self.assertTrue(torch.equal(value, fresh_model.node_category_head.state_dict()[key]))
            optimizer = torch.optim.Adam(fresh_model.parameters(),lr=.01,weight_decay=.1)
            inputs = self.inputs()
            x,adj,flags,u,state_t = inputs
            node,bond = fresh_model.categorical_attribute_inputs(flags,adj.bool(),torch.zeros((2,4),dtype=torch.long),torch.zeros((2,4,4),dtype=torch.long))
            output = fresh_model.forward_attributes(*inputs,node_attr=node,edge_attr=bond,attribute_time=torch.ones(2)*.5)
            loss = output["node_logits"].square().mean() + output["edge_logits"].square().mean()
            loss.backward();optimizer.step()
            proof = assert_frozen_topology(fresh_x,fresh_model,frozen)
            self.assertTrue(proof["topology_parameters_frozen"])
            self.assertTrue(any(not torch.equal(value, reference_model.state_dict()[key]) for key,value in fresh_model.state_dict().items() if key.startswith("attribute_")))
            with torch.no_grad():
                next(fresh_model.topology_parameters()).add_(1)
            with self.assertRaisesRegex(AssertionError,"changed frozen"):
                assert_frozen_topology(fresh_x,fresh_model,frozen)


class PopulationTests(unittest.TestCase):
    def fixture(self, root):
        import networkx as nx
        run=root/"reference";parent=run/"generations/pair";child=root/"control"
        parent.mkdir(parents=True); child.mkdir()
        train=run/"train"; (train/"checkpoints").mkdir(parents=True)
        checkpoint=train/"checkpoints/grapher.pt";checkpoint.write_bytes(b"trusted fixture")
        tm={"checkpoint":{"sha256":sha256(checkpoint)},"dataset":{"benchmark_id":"qm9","fingerprint":"split-hash"},"train_seed":42,"checkpoint_selection":{"epoch":200},"options":{"train":{"epochs":200},"sample":{"use_ema":False}}}
        (train/"manifest.json").write_text(json.dumps(tm))
        graphs=[]
        for index in range(2):
            graph=nx.path_graph(4);graph.graph["sample_id"]=f"42:{index}";graphs.append(graph)
        pairing={"checkpoint_sha256":sha256(checkpoint),"per_graph_sha256":["state0","state1"],"hash_inputs":["all states"],"batch_size":2,"generation_seed":42}
        for directory in (parent,child):
            files={}
            for name,data in (("topology_graphs",graphs),("base_graphs",graphs),("sampled_basis_indices",[2,5])):
                path=directory/(name+".pkl");path.write_bytes(pickle.dumps(data));files[name]={"path":path.name,"sha256":sha256(path)}
            (directory/"topology_pairing.json").write_text(json.dumps(pairing));files["topology_pairing"]={"path":"topology_pairing.json","sha256":sha256(directory/"topology_pairing.json")}
            (directory/"manifest.json").write_text(json.dumps({"checkpoint":{"path":str(checkpoint),"sha256":sha256(checkpoint)},"generation_seed":42,"num_generated":2,"num_requested":2,"effective_sample_config":{"use_ema":False,"separate_attribute_rng":True},**files}))
        return run,parent,child

    def test_missing_parent_fails_and_matched_topology_is_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(FileNotFoundError):
                require_attribute_reference(root,root,dataset="qm9",seed=42,n=2)
            run,parent,child=self.fixture(root)
            ref=require_attribute_reference(run,parent,dataset="qm9",seed=42,n=2)
            self.assertTrue(verify_attribute_population(child,ref)["fixed_topology_verified"])
            path=child/"topology_pairing.json"
            data=json.loads(path.read_text());data["per_graph_sha256"][0]="wrong-state";path.write_text(json.dumps(data))
            # Update the generation manifest, so failure proves cross-control
            # matching rather than merely internal artifact corruption detection.
            mp=child/"manifest.json";manifest=json.loads(mp.read_text());manifest["topology_pairing"]["sha256"]=sha256(path);mp.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,"topology evidence"):
                verify_attribute_population(child,ref)

    def test_reference_checkpoint_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run,parent,child=self.fixture(Path(temp))
            (run/"train/checkpoints/grapher.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"checkpoint hash"):
                require_attribute_reference(run,parent,dataset="qm9",seed=42,n=2)


if __name__ == "__main__":
    unittest.main()
