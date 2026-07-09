# GraphES: Coarse-to-Fine Graph Generation

GraphES implements a coarse-to-fine graph generation pipeline for generic graphs and molecular graphs.

The core idea is:

```text
training graphs
  -> permutation-invariant graph summaries
  -> learned summary generator
  -> coarse graph constructor
  -> GraphER rewiring refinement
  -> evaluation
```

For molecular generation, GraphES currently follows a topology-first design:

```text
molecular training graphs
  -> topology-only graph generation
  -> topology-conditioned molecular attribute generation
  -> molecular evaluation
```

---

## 1. Current scope

GraphES supports:

- generic graph generation:
  - SBM
  - grid
  - large grid
  - Citeseer ego networks
- QM9 topology-only graph generation
- QM9 attributed molecular graph generation
- empirical summary sampling
- learned SummaryVAE summary generation
- DegreeHistogramVAE for degree-sequence generation
- graphlet-history topology summaries
- Havel-Hakimi coarse graph construction
- GraphER-Opt energy-guided rewiring
- learned GraphER action-selector refinement
- topology-conditioned mixture CatFlow for molecular attributes
- molecular evaluation:
  - validity
  - uniqueness
  - novelty
  - NSPDK MMD
  - FCD

---

## 2. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

For ORCA orbit evaluation, set:

```bash
export ORCA_EXEC=/path/to/orca
```

---

## 3. Repository structure

```text
src/grapher/properties/summary.py
    permutation-invariant graph summaries and energy terms

src/grapher/properties/sampler.py
    empirical, learned, hybrid, and graphlet-history summary samplers

src/grapher/generators/degree_vae.py
    DegreeHistogramVAE for degree-sequence generation

src/grapher/generators/summary_vae.py
    SummaryVAE for learned graph summaries

src/grapher/generators/summary_vectorizer.py
    vectorization of graph summaries, including graphlet histories

src/grapher/construction/coarse.py
    Havel-Hakimi coarse graph construction

src/grapher/refinement/rewiring.py
    valid double-edge swap actions

src/grapher/refinement/grapher_opt.py
    training-free GraphER-Opt refinement

src/grapher/refinement/learned_selector.py
    learned GraphER action selector

src/grapher/evaluation/metrics.py
    generic graph metrics, including graphlet-history MMD

src/grapher/pipeline/coarse_to_fine.py
    end-to-end generic graph generation pipeline

src/grapher/molecular/
    QM9 topology preparation, molecular attribute generation, and molecular utilities
```

---

# Part A. Generic graph generation

## 4. Prepare generic datasets

Prepare SBM:

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset sbm \
  --root outputs/datasets
```

Other supported generic datasets:

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset grid \
  --root outputs/datasets

PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset grid_large \
  --root outputs/datasets

PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset ego \
  --root outputs/datasets
```

---

## 5. Baseline generic graph pipeline

The baseline generic pipeline uses:

```text
degree histogram
clustering histogram
spectral histogram
motif / orbit summaries
```

Example config:

```text
configs/experiments/sbm_report_degreevae.yaml
```

Train the degree generator:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --output-dir outputs/degree_generators/sbm_report \
  --epochs 300 \
  --batch-size 32 \
  --beta 0.005 \
  --degree-weight 5.0 \
  --edge-moment-weight 0.1 \
  --seed 42
```

Build the GraphER teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --output-dir outputs/teachers/sbm_report \
  --num-trajectories 512 \
  --seed 42 \
  --debug
```

Train the learned rewiring selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --teacher-dir outputs/teachers/sbm_report \
  --output-dir outputs/selectors/sbm_report \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

Generate graphs:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --num-generate 40 \
  --output-dir outputs/coarse_to_fine/sbm_report_degreevae_seed42 \
  --debug
```

The output should include:

```text
coarse
learned_selector
grapher_opt
random_rewire
```

---

# Part B. Graphlet-history topology summaries

## 6. Graphlet-history design

The graphlet-history Stage 1 generator uses:

```text
degree histogram
+
graphlet history
```

The graphlet history is a sequence of graphlet-frequency vectors:

```text
h_3(G), h_4(G), ..., h_K(G)
```

For the first implementation, use:

```text
K = 5
connected induced graphlets only
sampled graphlets for larger generic graphs
exact graphlets for small molecular graphs
```

The generic graphlet-history config is:

```text
configs/experiments/sbm_report_graphlet_history.yaml
```

Important config block:

```yaml
summary:
  degree_hist_max_degree: auto
  motif_proxy: false
  orbit_count: false
  graphlet_history: true
  graphlet_k_min: 3
  graphlet_k_max: 5
  graphlet_connected_only: true
  graphlet_num_samples: 2048

energy:
  degree_weight: 0.0
  clustering_weight: 0.0
  spectral_weight: 0.0
  motif_weight: 0.0
  triangle_weight: 0.0
  graphlet_weight: 1.0
  graphlet_size_weights:
    "3": 1.0
    "4": 1.0
    "5": 1.0
  normalize_terms: true
```

For SBM, `graphlet_num_samples: 2048` is recommended because exact graphlet enumeration up to size 5 is expensive.

---

## 7. Run graphlet-history pipeline on SBM

Run the graphlet-history tests:

```bash
PYTHONPATH=src pytest tests/test_graphlet_history_summary.py -q
```

Train graphlet-history SummaryVAE:

```bash
PYTHONPATH=src python scripts/train_summary_generator.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --output-dir outputs/summary_generators/sbm_graphlet_history \
  --epochs 300 \
  --batch-size 32 \
  --beta 0.005 \
  --seed 42
```

Train DegreeVAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --output-dir outputs/degree_generators/sbm_graphlet_history \
  --epochs 300 \
  --batch-size 32 \
  --beta 0.005 \
  --degree-weight 5.0 \
  --edge-moment-weight 0.1 \
  --seed 42
```

Verify degree generation:

```bash
PYTHONPATH=src python scripts/verify_degree_generator.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --num-samples 1000
```

Run a small teacher-cache smoke test:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --output-dir outputs/teachers/sbm_graphlet_history_smoke \
  --num-trajectories 20 \
  --seed 42 \
  --debug
```

Build the full teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --output-dir outputs/teachers/sbm_graphlet_history \
  --num-trajectories 512 \
  --seed 42 \
  --debug
```

Train the learned selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --teacher-dir outputs/teachers/sbm_graphlet_history \
  --output-dir outputs/selectors/sbm_graphlet_history \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

Generate graphs:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --num-generate 40 \
  --output-dir outputs/coarse_to_fine/sbm_graphlet_history_seed42 \
  --debug
```

Main success conditions:

```text
learned_selector graphlet MMD < coarse graphlet MMD
learned_selector graphlet MMD < random_rewire graphlet MMD
connectedness = 1.0
degree MMD is preserved across refinement rows
```

---

## 8. Compare old summary vs graphlet-history summary

Run the old degree-summary baseline:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_report_degreevae.yaml \
  --num-generate 40 \
  --output-dir outputs/coarse_to_fine/sbm_old_summary_seed42 \
  --debug
```

Run the graphlet-history model:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --num-generate 40 \
  --output-dir outputs/coarse_to_fine/sbm_graphlet_history_seed42 \
  --debug
```

Compare:

```text
degree MMD
graphlet-history MMD
clustering MMD
spectral MMD
motif MMD
orbit MMD
connectedness
```

Expected behavior:

```text
graphlet-history summary should improve graphlet, motif, orbit, and possibly clustering metrics
spectral MMD may require an additional spectral summary if it degrades
```

---

# Part C. QM9 topology-first molecular generation

## 9. Prepare QM9 topology and attributed datasets

Download QM9 SDF:

```bash
mkdir -p data/qm9_deepchem
cd data/qm9_deepchem

wget -O qm9.tar.gz \
  https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/qm9.tar.gz

tar -xzf qm9.tar.gz

find . -maxdepth 2 -type f | grep -E "qm9.sdf|sdf$"
cd ../..
```

Prepare topology-only and attributed QM9 datasets:

```bash
PYTHONPATH=src python scripts/prepare_qm9_topology_dataset.py \
  --source sdf \
  --sdf-file data/qm9_deepchem/qm9.sdf \
  --root outputs/datasets
```

This creates:

```text
outputs/datasets/qm9_topology/
outputs/datasets/qm9_attributed/
```

`qm9_topology` contains unlabeled heavy-atom topologies.

`qm9_attributed` contains atom and bond labels.

---

## 10. QM9 topology generation with original topology summaries

Use:

```text
configs/experiments/qm9_topology_mixture_catflow.yaml
```

Train topology SummaryVAE:

```bash
PYTHONPATH=src python scripts/train_summary_generator.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --output-dir outputs/summary_generators/qm9_topology \
  --epochs 300 \
  --batch-size 64 \
  --beta 0.005 \
  --seed 42
```

Train topology DegreeVAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --output-dir outputs/degree_generators/qm9_topology \
  --epochs 300 \
  --batch-size 64 \
  --beta 0.005 \
  --degree-weight 5.0 \
  --edge-moment-weight 0.1 \
  --seed 42
```

Build topology teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --output-dir outputs/teachers/qm9_topology \
  --num-trajectories 512 \
  --seed 42 \
  --debug
```

Train topology selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --teacher-dir outputs/teachers/qm9_topology \
  --output-dir outputs/selectors/qm9_topology \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

Generate QM9 topologies:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --num-generate 10000 \
  --output-dir outputs/coarse_to_fine/qm9_topology_seed42 \
  --debug
```

Generated topologies are saved to:

```text
outputs/coarse_to_fine/qm9_topology_seed42/learned_selector_graphs.pkl
```

---

## 11. QM9 topology generation with graphlet-history summaries

Use:

```text
configs/experiments/qm9_topology_graphlet_history.yaml
```

Train graphlet-history topology SummaryVAE:

```bash
PYTHONPATH=src python scripts/train_summary_generator.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml \
  --output-dir outputs/summary_generators/qm9_topology_graphlet \
  --epochs 300 \
  --batch-size 64 \
  --beta 0.005 \
  --seed 42
```

Train topology DegreeVAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml \
  --output-dir outputs/degree_generators/qm9_topology_graphlet \
  --epochs 300 \
  --batch-size 64 \
  --beta 0.005 \
  --degree-weight 5.0 \
  --edge-moment-weight 0.1 \
  --seed 42
```

Build topology teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml \
  --output-dir outputs/teachers/qm9_topology_graphlet \
  --num-trajectories 512 \
  --seed 42 \
  --debug
```

Train topology selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml \
  --teacher-dir outputs/teachers/qm9_topology_graphlet \
  --output-dir outputs/selectors/qm9_topology_graphlet \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

Generate graphlet-history QM9 topologies:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml \
  --num-generate 10000 \
  --output-dir outputs/coarse_to_fine/qm9_topology_graphlet_seed42 \
  --debug
```

Generated topologies are saved to:

```text
outputs/coarse_to_fine/qm9_topology_graphlet_seed42/learned_selector_graphs.pkl
```

---

# Part D. QM9 molecular attribute generation

## 12. Train topology-conditioned mixture CatFlow

Train a topology-conditioned mixture CatFlow model for atom and bond labels:

```bash
PYTHONPATH=src python scripts/train_qm9_mixture_catflow.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --output-dir outputs/attribute_flows/qm9_mixture_catflow \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

A CatFlow-style Stage 2 ablation is obtained by setting:

```yaml
num_mixtures: 1
```

in:

```text
configs/experiments/qm9_topology_catflow_stage2_ablation.yaml
```

---

## 13. Generate molecular graphs from generated topologies

Using original topology output:

```bash
PYTHONPATH=src python scripts/sample_qm9_mixture_catflow.py \
  --checkpoint outputs/attribute_flows/qm9_mixture_catflow/checkpoint.pt \
  --topology-graphs outputs/coarse_to_fine/qm9_topology_seed42/learned_selector_graphs.pkl \
  --output-dir outputs/molecular/qm9_topology_first_mixture_catflow \
  --steps 64 \
  --temperature 1.0 \
  --seed 42
```

Using graphlet-history topology output:

```bash
PYTHONPATH=src python scripts/sample_qm9_mixture_catflow.py \
  --checkpoint outputs/attribute_flows/qm9_mixture_catflow/checkpoint.pt \
  --topology-graphs outputs/coarse_to_fine/qm9_topology_graphlet_seed42/learned_selector_graphs.pkl \
  --output-dir outputs/molecular/qm9_graphlet_topology_first_mixture_catflow \
  --steps 64 \
  --temperature 1.0 \
  --seed 42
```

---

## 14. Evaluate generated molecules

Evaluate generated molecules:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/molecular/qm9_topology_first_mixture_catflow \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/molecular/qm9_topology_first_mixture_catflow/evaluation
```

The evaluator reports:

```text
num_generated_graphs
num_valid_generated_molecules
num_invalid_generated_molecules
validity_without_correction
uniqueness_rate
unique_valid_count
novelty_rate
novel_unique_valid_count
NSPDK MMD
NSPDK MMD valid only
FCD
```

If FCD dependencies are unavailable:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/molecular/qm9_topology_first_mixture_catflow \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/molecular/qm9_topology_first_mixture_catflow/evaluation \
  --skip-fcd
```

---

## 15. Extract attribute-related topology summaries

Attribute-related topology summaries can be extracted with:

```bash
PYTHONPATH=src python scripts/extract_qm9_molecular_summaries.py \
  --dataset qm9_attributed \
  --root outputs/datasets \
  --output-dir outputs/molecular_summaries/qm9
```

These summaries include atom histograms, typed degree information, bond histograms, and atom-pair compatibility statistics. They are useful for future chemistry-aware topology generation.

---

# Part E. Diagnostics

## 16. Check generated topology uniqueness

Use this to check whether repeated molecules come from repeated generated topologies:

```bash
PYTHONPATH=src python - <<'PY'
import pickle
from collections import Counter
import networkx as nx

path = "outputs/coarse_to_fine/qm9_topology_seed42/learned_selector_graphs.pkl"

with open(path, "rb") as f:
    graphs = pickle.load(f)

hashes = [
    nx.weisfeiler_lehman_graph_hash(g, node_attr=None, edge_attr=None, iterations=3)
    for g in graphs
]

cnt = Counter(hashes)

print("num_topologies:", len(graphs))
print("unique_topologies:", len(cnt))
print("topology_uniqueness:", len(cnt) / max(len(graphs), 1))
print("top repeated topology counts:")
for h, c in cnt.most_common(20):
    print(c, h)
PY
```

---

## 17. Check repeated generated molecules

```bash
python - <<'PY'
from collections import Counter

path = "outputs/molecular/qm9_topology_first_mixture_catflow/evaluation/valid_generated.smi"

smiles = []
with open(path) as f:
    for line in f:
        s = line.strip().split()[0]
        if s:
            smiles.append(s)

cnt = Counter(smiles)

print("valid smiles:", len(smiles))
print("unique smiles:", len(cnt))
print("uniqueness:", len(cnt) / max(len(smiles), 1))
print("top repeated molecules:")
for s, c in cnt.most_common(30):
    if c > 1:
        print(c, s)
PY
```

---

## 18. Compare Stage 2 on real vs generated topologies

This isolates whether molecular FCD is limited by the topology generator or the molecular attribute generator.

Generate attributes on real test topologies:

```bash
PYTHONPATH=src python scripts/sample_qm9_mixture_catflow.py \
  --checkpoint outputs/attribute_flows/qm9_mixture_catflow/checkpoint.pt \
  --topology-graphs outputs/datasets/qm9_topology/test.pkl \
  --output-dir outputs/molecular/qm9_real_test_topology_stage2 \
  --steps 64 \
  --temperature 1.0 \
  --seed 42
```

Evaluate:

```bash
PYTHONPATH=src python scripts/evaluate_generated_molecules.py \
  --generated-dir outputs/molecular/qm9_real_test_topology_stage2 \
  --dataset-root outputs/datasets \
  --dataset qm9_attributed \
  --reference-split test \
  --train-split train \
  --output-dir outputs/molecular/qm9_real_test_topology_stage2/evaluation
```

Interpretation:

```text
real-topology Stage 2 good, generated-topology Stage 2 bad:
    topology generator is the bottleneck

real-topology Stage 2 still bad:
    attribute flow is the bottleneck
```

---

# Part F. Useful configs

```text
configs/experiments/sbm_report_degreevae.yaml
    generic graph pipeline with degree generator and original structural summaries

configs/experiments/sbm_report_graphlet_history.yaml
    generic graph pipeline with degree generator and graphlet-history summaries

configs/experiments/qm9_topology_mixture_catflow.yaml
    QM9 topology-first molecular pipeline

configs/experiments/qm9_topology_catflow_stage2_ablation.yaml
    Stage 2 CatFlow ablation with num_mixtures = 1

configs/experiments/qm9_joint_catflow_baseline.yaml
    joint dense CatFlow-style baseline

configs/experiments/qm9_topology_graphlet_history.yaml
    QM9 topology generator with degree histogram and graphlet-history summaries
```

---

# Part G. Expected output locations

Generic graph outputs:

```text
outputs/coarse_to_fine/<run_name>/
  coarse_graphs.pkl
  learned_selector_graphs.pkl
  grapher_opt_graphs.pkl
  random_rewire_graphs.pkl
  metrics.json
  sampled_summaries.json
```

Degree generator outputs:

```text
outputs/degree_generators/<run_name>/
  checkpoint.pt
  degree_vectorizer.json
  training_report.json
```

Summary generator outputs:

```text
outputs/summary_generators/<run_name>/
  checkpoint.pt
  summary_vectorizer.json
  training_report.json
```

Teacher cache outputs:

```text
outputs/teachers/<run_name>/
  train.jsonl
  val.jsonl
  teacher_report.json
```

Selector outputs:

```text
outputs/selectors/<run_name>/
  checkpoint.pt
  training_report.json
```

Molecular outputs:

```text
outputs/molecular/<run_name>/
  molecular_graphs.pkl
  generated.smi
  evaluation/
    molecular_evaluation_metrics.json
    valid_generated.smi
```

---

# Part H. Recommended experiment order

## Generic graph experiment

```text
1. Prepare SBM.
2. Train graphlet-history SummaryVAE.
3. Train DegreeVAE.
4. Verify degree generator.
5. Build a small teacher cache as smoke test.
6. Build full teacher cache.
7. Train learned selector.
8. Generate graphs.
9. Compare against old summary baseline.
10. Repeat for seeds 42, 43, and 44.
```

## QM9 experiment

```text
1. Prepare QM9 topology and attributed splits.
2. Train topology generator.
3. Generate QM9 topologies.
4. Train topology-conditioned mixture CatFlow.
5. Generate molecules.
6. Evaluate validity, uniqueness, novelty, NSPDK, and FCD.
7. Compare real-topology Stage 2 against generated-topology Stage 2.
8. Compare original topology summaries against graphlet-history topology summaries.
```