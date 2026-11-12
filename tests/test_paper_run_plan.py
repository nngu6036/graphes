"""Integration tests for the runbook dispatcher and missing-cell coverage."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('paper_plan',ROOT/'scripts/run_grapher_paper_experiments.py')
plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)


class RunPlanTests(unittest.TestCase):
    def preview(self,*argv):
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.assertEqual(plan.main(list(argv)),0)
        return stream.getvalue()

    def test_all_experiments_preview_without_data_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination=Path(tmp)/'not_created'
            for experiment in plan.EXPERIMENTS:
                with self.subTest(experiment=experiment):
                    result=self.preview(experiment,'--seeds','41','--report-root',str(destination))
                    self.assertIn('PREVIEW ONLY',result)
                    self.assertFalse(destination.exists())

    def test_temperature_has_saved_d_and_dr_for_each_value(self):
        text=self.preview('temperature','--datasets','ego_small','--seeds','41')
        self.assertIn('--temperature-rewire-budget 1:8',text)
        for tau in ('0','0p1','0p2','0p4','0p8'):
            self.assertIn('/D_tau'+tau+'/base_graphs.pkl',text)
            self.assertIn('/DR_tau'+tau+'/base_graphs.pkl',text)

    def test_budgets_cover_both_decoders(self):
        text=self.preview('budgets','--datasets','community_small','--seeds','41')
        for decoder in ('T','D'):
            for budget in ('0','1','1to2','1to8'):
                self.assertIn('/'+decoder+'_B'+budget+'/base_graphs.pkl',text)

    def test_learning_summary_attribute_reference(self):
        text=self.preview('learning_summary','--datasets','qm9','--seeds','41')
        self.assertIn('A1_minus_A0=A1,A0',text)
        self.assertIn('A2_minus_A0=A2,A0',text)
        self.assertNotIn(',S3=',text)
        self.assertIn('--pairing seed',text)

    def test_runtime_distinct_variants_and_new_ids(self):
        text=self.preview('runtime','--datasets','community_small','qm9','--seeds','41')
        for name in ('community_small_TR','community_small_DR','qm9_T','qm9_TR'):
            self.assertIn('--label '+name+'_seed_41',text)
        self.assertIn('paper_cost_v1_TR_seed_41_n_1024',text)
        self.assertNotIn('--overwrite',text)

    def test_invalid_molecular_exact_degree_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.preview('temperature','--datasets','qm9')

    def test_missing_cell_inventory_is_complete_and_noop_explicit(self):
        rows=json.loads((ROOT/'docs/paper_missing_cells.json').read_text())
        self.assertEqual(len(rows),254)
        skipped=[r for r in rows if r['command'] is None]
        self.assertEqual(len(skipped),5)
        self.assertTrue(all(r['dataset']=='QM9' and r['variant_or_seed'].startswith('S3') for r in skipped))
        self.assertEqual(len({r['id'] for r in rows}),254)

    def test_baseline_defaults_use_supported_native_configs(self):
        text=self.preview('baseline','--datasets','zinc','--seeds','41','--baseline-stage','generate')
        for model in plan.MOLECULAR_BASELINES:
            self.assertIn('run_'+model+'_baseline.py',text)
            self.assertTrue((ROOT/f'configs/baselines/{model}_zinc.yaml').is_file())
        self.assertNotIn('run_graphrnn',text)


if __name__=='__main__':
    unittest.main()
