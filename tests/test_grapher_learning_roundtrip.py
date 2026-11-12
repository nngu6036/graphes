"""End-to-end CPU fixture; four small 200-update runs, no benchmark metrics.

This verifies the real wrapper train/save/load/generate path and exact population
matching for A0/A1/A2. The fixture contains only two synthetic four-node graphs.
"""
from __future__ import annotations
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("networkx"), "torch/networkx required")
class LearningRoundtripTests(unittest.TestCase):
    def test_frozen_topology_survives_real_training_generation(self):
        import copy, json, os, pickle, sys, tempfile
        from pathlib import Path
        import networkx as nx
        import torch
        from grapher.models.base import DatasetReference, RunSpec, TrainRequest, GenerateRequest
        from grapher.models.grapher.wrapper import GraphERWrapper
        from grapher.models.grapher.hybrid_attributed_loggap import default_options
        from grapher.experiments.learning_ablation import require_attribute_reference, verify_attribute_population

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="grapher_learning_fixture_") as directory, contextlib.redirect_stdout(io.StringIO()):
            os.environ['GRAPHER_EIGH_BACKEND'] = 'cpu'
            root=Path(directory)
            dataset_root=root/'datasets';(dataset_root/'qm9_attributed').mkdir(parents=True)
            graphs=[]
            for graph in (nx.path_graph(4), nx.cycle_graph(4)):
             nx.set_node_attributes(graph,6,'atomic_num');nx.set_edge_attributes(graph,1,'bond_type');graphs.append(graph)
            for split in ('train','val','test'):
             (dataset_root/'qm9_attributed'/(split+'.pkl')).write_bytes(pickle.dumps(graphs))
            opts=default_options()
            opts['train'].update(epochs=200,batch_size=2,lr=.001,node_lr=.001,spectrum_lr=.001,log_every=200,validation_every=200)
            opts['model'].update(max_nodes=4,max_feat_num=4,hidden_dim=4,depth=1,ppgn_hidden_dim=4,ppgn_depth=1)
            opts['structural_features']={'enabled':False}
            opts['preprocess'].update(num_workers=0,cache_structure_targets=False)
            opts['sde']['x'].update(num_scales=2,beta_min=.01,beta_max=.05)
            opts['sde']['spectrum'].update(num_scales=2,beta_min=.01,beta_max=.05)
            opts['sample'].update(use_ema=False,separate_attribute_rng=True)
            opts['topology_summary']['graphlet']['orders']=[3]
            opts['attribute_summary']['typed_graphlet'].update(orders=[3],max_basis_graphs=20)
            opts['attribute_summary']['typed_degree']['enabled']=False
            opts['topology_summary']['degree']['enabled']=False
            opts['attributed'].update(node_categories=[6,8],edge_categories=[1,2])
            opts['attributed']['decode']['constraint_mode']='none'
            opts['attribute_diffusion'].update(steps=2,sample_steps=2)
            opts['generation_batch_size']=2
            opts['runtime']={'device':'cpu'}
            wrapper=GraphERWrapper();dataset=DatasetReference('qm9',root=dataset_root,serialized_id='qm9_attributed')
            ref_run=RunSpec.for_seed(model_id='grapher',dataset_id='qm9',seed=42,output_root=root/'outputs',run_id='reference')
            training=wrapper.train(TrainRequest(run=ref_run,dataset=dataset,options=opts))
            generated=wrapper.generate(GenerateRequest(run=ref_run,checkpoint_path=training.checkpoint_path,num_graphs=2,generation_seed=42,generation_id='parent',options={'runtime':{'device':'cpu'}}))
            ref=require_attribute_reference(ref_run.layout.run_dir,generated.generation_dir,dataset='qm9',seed=42,n=2)
            results=[]
            for variant in ('A0','A1','A2'):
             control=copy.deepcopy(opts);control['train']['attribute_only']=True;control['train']['frozen_topology_reference']=ref
             if variant=='A1': control['attribute_summary']['loss_weight']=0.0
             if variant=='A2': control['attributed']['spectral_context']='none'
             run=RunSpec.for_seed(model_id='grapher',dataset_id='qm9',seed=42,output_root=root/'outputs',run_id=variant)
             trained=wrapper.train(TrainRequest(run=run,dataset=dataset,options=control))
             gen=wrapper.generate(GenerateRequest(run=run,checkpoint_path=trained.checkpoint_path,num_graphs=2,generation_seed=42,generation_id='control',options={'runtime':{'device':'cpu'}}))
             proof=verify_attribute_population(gen.generation_dir,ref)
             manifest=json.loads(trained.manifest_path.read_text())
             assert manifest['learning_ablation']['topology_parameter_sha256_before']==manifest['learning_ablation']['topology_parameter_sha256_after']
             results.append({'variant':variant,'proof':proof,'train_epoch':manifest['checkpoint_selection']['epoch']})
            self.assertEqual(len(results), 3)

if __name__ == "__main__":
    unittest.main()
