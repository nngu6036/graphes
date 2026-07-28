# GraphES: Target-Conditioned Graph Refinement

GraphES implements a stochastic, target-summary-conditioned, constraint-preserving graph refinement generator.

For a generic graph, the revised pipeline is:

```text
training graphs
  -> degree-sequence generator p(D)
  -> connected Havel-Hakimi realization G0(D)
  -> conditional target-summary generator p(s* | D, z)
  -> valid double-edge-swap refinement toward s*
  -> evaluation
```

The target is a structural description—initially graphlet histories and clustering statistics—not a sampled adjacency matrix. This keeps the target compatible with the fixed-degree rewiring state space and avoids moving the full combinatorial graph-generation problem into the estimator.

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
- legacy unconditional SummaryVAE generation
- degree-conditioned target-summary CVAE generation
- DegreeHistogramVAE for degree-sequence generation
- graphlet-history topology summaries
- Havel-Hakimi coarse graph construction
- GraphER-Opt energy-guided rewiring
- learned GraphER action-selector refinement with:
  - current/target/residual summary features
  - soft energy-based teacher distributions
  - explicit `STOP` supervision
  - optional neural top-\(K\) plus exact-energy selection
  - DAgger-style on-policy teacher aggregation
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

Install a PyTorch build appropriate for the machine separately if it is not already available. CUDA is recommended for training; CPU execution is supported for tests and small smoke runs.

Set `PYTHONPATH`:

```bash
export PYTHONPATH=src
```

For ORCA orbit evaluation, set:

```bash
export ORCA_EXEC=/path/to/orca
```

For fast graphlet canonicalization, install nauty/Traces and set:

```bash
export NAUTY_EXEC=/path/to/labelg
```

Without `labelg`, topology graphlets of size at most 8 use an exact Python fallback. The fallback is suitable for tests and small experiments but is much slower than nauty.

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
    legacy SummaryVAE, conditional target-summary CVAE, and vectorization

src/grapher/construction/coarse.py
    Havel-Hakimi coarse graph construction

src/grapher/refinement/rewiring.py
    valid double-edge swap actions

src/grapher/refinement/grapher_opt.py
    training-free GraphER-Opt refinement

src/grapher/refinement/learned_selector.py
    learned selector, learned STOP, and hybrid neural/exact refinement

src/grapher/refinement/features.py
    permutation-invariant current/target/residual and action features

src/grapher/evaluation/metrics.py
    generic graph metrics, including graphlet-history MMD

src/grapher/evaluation/degree_sequences.py
    DH-VAE degree-distribution, diversity, and size-fidelity metrics

src/grapher/pipeline/coarse_to_fine.py
    end-to-end generic graph generation pipeline

src/grapher/molecular/
    QM9 topology preparation, molecular attribute generation, and molecular utilities

scripts/verify_target_summary_generator.py
    held-out conditional-summary reconstruction and diversity checks

configs/experiments/sbm_target_refinement.yaml
    complete configuration for the revised generic-graph proposal
```

---

# Part A. Revised target-conditioned GraphER

## 4. Model contract

The generator factorizes as

\[
p(G)=\sum_D p_\psi(D)\int p(z)\,
p_\phi(\mathbf{s}^{*}\mid D,z)\,
p_\theta(G\mid G_{\mathrm{HH}}(D),D,\mathbf{s}^{*})\,dz.
\]

Each component has one responsibility:

| Component | Responsibility |
| --- | --- |
| `DegreeHistogramVAE` | Samples \(D\), which fixes graph size, edge count, and degree multiset |
| Havel-Hakimi constructor | Creates a simple connected starting graph \(G_0\) with exactly \(D\) |
| `ConditionalSummaryVAE` | Samples a structural mode \(\mathbf{s}^{*}\) conditioned on \(D\) and latent \(z\) |
| Energy teacher | Labels valid swap candidates by exact reduction in summary discrepancy |
| Learned selector | Ranks a finite valid candidate set and optionally predicts `STOP` |
| Hybrid refiner | Re-evaluates the neural top-\(K\) candidates with exact energy before applying one |

The sampled target summary remains fixed for one rollout. Degree and density have zero refinement-energy weight because a double-edge swap cannot change them.

The default target summary in `sbm_target_refinement.yaml` is

\[
\mathbf{s}(G)=
\left[\mathbf{h}_3(G),\mathbf{h}_4(G),\mathbf{c}(G)\right],
\]

where \(\mathbf{h}_k\) is the connected induced graphlet-frequency vector and \(\mathbf{c}\) is the clustering-coefficient histogram.

## 5. Complete step-by-step training curriculum

Run every command from the repository root.

### Step 0: set up and test the repository

```bash
export PYTHONPATH=src
PYTHONPATH=src pytest -q
```

The expected result for this release is `18 passed`.

### Step 1: prepare the training graphs

```bash
PYTHONPATH=src python scripts/prepare_generic_dataset.py \
  --dataset sbm \
  --root outputs/datasets
```

This creates the train/validation/test splits consumed by every later stage.

### Step 2: train the degree-sequence generator

The corrected DH-VAE implements the paper's size-conditioned factorization
\(p_\theta(D\mid n,z)\). The decoder receives a continuous embedding of the
true graph size during training and of the sampled graph size during
generation. The degree-moment loss is calculated from the decoded categorical
distribution itself, so it directly constrains the implied edge count.

Old checkpoints are incompatible because their decoder implements
\(p_\theta(D\mid z)\). Remove or rename the old checkpoint, then retrain:

```bash
mv outputs/degree_generators/sbm_target_refinement/checkpoint.pt \
  outputs/degree_generators/sbm_target_refinement/checkpoint_unconditional.pt

PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/sbm_target_refinement.yaml
```

Output:

```text
outputs/degree_generators/sbm_target_refinement/checkpoint.pt
```

Validate the graphicality, connected-feasibility, node-count distribution, and degree MMD before proceeding. Refinement cannot repair a wrong degree sequence.

### Step 3: evaluate the degree-sequence generator

```bash
PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/sbm_target_refinement.yaml
```

Outputs:

```text
outputs/degree_generators/sbm_target_refinement/evaluation/degree_evaluation.json
outputs/degree_generators/sbm_target_refinement/evaluation/generated_degree_sequences.json
```

The report contains four distribution comparisons using the same held-out test
split and the same RBF-kernel bandwidth:

```text
Train <-> Test
Posterior reconstruction -> Test
Raw prior -> Test
Accepted prior -> Test
```

`Posterior reconstruction` decodes the test histograms through
\(q_\phi(z\mid h_D,n)\) using the posterior mean and performs no repair.
`Raw prior` evaluates the first multinomial draw from \(z\sim N(0,I)\).
`Accepted prior` evaluates the sequence after genuine rejection sampling and,
only if the retry budget is exhausted, repair or empirical fallback.

For the manuscript table, use the accepted-prior `degree_kl` and `degree_mmd`.
The KL direction is
explicitly fixed as \(D_{\mathrm{KL}}(P_{\mathrm{test}}\Vert
P_{\mathrm{candidate}})\), and MMD is computed over per-graph normalized
degree histograms. `Train <-> Test` is the empirical oracle-gap reference, not
a model.

Inspect both native decoder quality and final accepted quality:

- `raw_graphicality_rate` and `raw_connected_feasible_rate` are measured on
  the first prior draw before repair;
- `accepted_without_postprocessing` is represented by the aggregate native
  acceptance rate in the saved per-sample diagnostics;
- `repair_usage_rate`, `mean_repair_l1_adjustment`, and
  `fallback_usage_rate` reveal how strongly sampling depends on
  post-processing;
- `accepted_graphicality_rate`,
  `accepted_connected_feasible_rate`, and `constructor_success_rate` should
  be `1.0`;
- `node_count_total_variation` and `edge_count_total_variation` measure size
  and edge-count fidelity;
- `sequence_uniqueness_rate`, `sequence_novelty_rate`, and
  `reference_sequence_coverage_rate` measure diversity and mode coverage.

Do not proceed merely because accepted graphicality is `1.0`: repair makes
that a hard guarantee. A healthy checkpoint should also have low KL/MMD
relative to the Train-Test oracle gap, low fallback usage, and non-trivial
sequence diversity.

Interpret the three model rows as follows:

- high posterior MMD means the conditional decoder or reconstruction loss is
  inadequate;
- low posterior MMD but high raw-prior MMD indicates posterior-prior mismatch;
- low raw-prior MMD but high accepted-prior MMD means post-processing is
  distorting the distribution.

For a quick smoke evaluation:

```bash
PYTHONPATH=src python scripts/evaluate_degree_generator.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --num-samples 64 \
  --max-reference-sequences 64
```

### Step 4: train \(p_\phi(\mathbf{s}^{*}\mid D,z)\)

```bash
PYTHONPATH=src python scripts/train_summary_generator.py \
  --config configs/experiments/sbm_target_refinement.yaml
```

The config sets `conditional_on_degree: true`. The encoder observes the training summary and degree condition; the decoder receives the condition and latent sample. Node count and degree losses are zero because those invariant fields are copied exactly from \(D\) during sampling.

Output:

```text
outputs/target_summary_generators/sbm_target_refinement/checkpoint.pt
```

Verify the checkpoint:

```bash
PYTHONPATH=src python scripts/verify_target_summary_generator.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --output outputs/target_summary_generators/sbm_target_refinement/verification.json
```

The required invariant check is:

```text
degree_condition_match_rate: 1.0
```

Also inspect reconstruction error and `prior_structural_diversity`. Near-zero diversity suggests posterior collapse or an overly large KL weight.

### Step 5: build oracle-guided teacher trajectories

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --target-source oracle \
  --output-dir outputs/teachers/sbm_target_refinement/oracle
```

For each training target, the script:

1. extracts its degree sequence and target summary;
2. constructs the connected Havel-Hakimi source graph;
3. samples a finite set of valid degree-preserving swaps;
4. computes each candidate's exact energy reduction;
5. saves a soft teacher distribution over candidates plus `STOP`;
6. applies an improving teacher action and repeats until convergence or the maximum budget.

Every cache record contains the current graph, current summary, fixed target summary, residual-compatible features, candidate swaps, energy changes, soft labels, and stopping label. The target adjacency matrix is never used to create a candidate.

### Step 6: train the oracle-guided selector

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --teacher-dir outputs/teachers/sbm_target_refinement/oracle \
  --output-dir outputs/selectors/sbm_target_refinement/oracle
```

The selector minimizes soft cross-entropy against the teacher distribution, with an optional Huber regression term on normalized energy improvement. Its version-2 features are permutation-invariant and include current summaries, target summaries, their residuals, invariant local swap features, and a `STOP` flag.

### Step 7: expose the selector to predicted targets

Build a deployment-aware cache using prior samples from the conditional target generator:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --target-source predicted \
  --output-dir outputs/teachers/sbm_target_refinement/predicted
```

Fine-tune from the oracle-guided checkpoint:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --teacher-dir outputs/teachers/sbm_target_refinement/predicted \
  --output-dir outputs/selectors/sbm_target_refinement/predicted \
  --resume-checkpoint outputs/selectors/sbm_target_refinement/oracle/checkpoint.pt
```

This stage removes the train/generation mismatch while keeping the target-summary model frozen.

### Step 8: aggregate on-policy states

Visit states generated by the current selector, but label every visited state with the exact energy teacher:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --target-source predicted \
  --rollout-selector-checkpoint outputs/selectors/sbm_target_refinement/predicted/checkpoint.pt \
  --output-dir outputs/teachers/sbm_target_refinement/on_policy
```

Fine-tune again:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --teacher-dir outputs/teachers/sbm_target_refinement/on_policy \
  --output-dir outputs/selectors/sbm_target_refinement/final \
  --resume-checkpoint outputs/selectors/sbm_target_refinement/predicted/checkpoint.pt
```

Repeat Step 8 if validation rollouts continue to improve. This is a DAgger-style loop: the learned selector chooses which states are visited, while exact metric evaluation supplies the labels.

### Step 9: generate and evaluate graphs

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_target_refinement.yaml \
  --selector-checkpoint outputs/selectors/sbm_target_refinement/final/checkpoint.pt \
  --num-generate 40 \
  --output-dir outputs/coarse_to_fine/sbm_target_refinement_seed42 \
  --debug
```

Generation performs:

```text
D ~ p_psi(D)
G0 = connected_havel_hakimi(D)
s* ~ p_phi(s | D, z)
repeat:
    sample valid double-edge swaps
    rank them with the learned selector
    compare STOP with the candidate scores
    exactly evaluate the neural top-K
    apply the best improving candidate
until STOP, convergence, patience exhaustion, or T_max
```

The final report compares:

- unrefined Havel-Hakimi graphs;
- the learned/hybrid selector;
- exact GraphER-Opt;
- random valid rewiring.

Check that connectivity and degree preservation are exactly `1.0`, and compare graphlet, clustering, and spectral MMD. `T_max` is only a hard compute budget; `STOP`, target-energy threshold, minimum improvement, and patience provide adaptive termination.

## 6. Recommended ablations

Run the following order so error sources remain identifiable:

1. oracle \(D\) + oracle summary + exact candidate search;
2. oracle \(D\) + oracle summary + learned selector;
3. oracle \(D\) + predicted summary + learned selector;
4. generated \(D\) + predicted summary + learned selector;
5. neural-only selection versus exact search versus neural top-\(K\) plus exact energy;
6. fixed step budget versus learned/adaptive stopping;
7. deterministic versus randomly relabeled Havel-Hakimi initialization.

For a pure metric-search baseline, set:

```yaml
refiner:
  type: grapher_opt
```

For neural-only selection, set `exact_top_k: 0`. For the recommended hybrid, keep `exact_top_k: 8`.

---

# Legacy generic graph generation

## Prepare generic datasets

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
configs/experiments/sbm_target_refinement.yaml
    revised degree-conditioned target-summary and GraphER refinement curriculum

configs/experiments/sbm_report.yaml
```

Train the degree generator:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/degreevae.yaml \
  --dataset sbm
```

Build the GraphER teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_report.yaml
```

Train the learned rewiring selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_report.yaml
```

Generate graphs:

```bash
PYTHONPATH=src python scripts/run_coarse_to_fine.py \
  --config configs/experiments/sbm_report.yaml \
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
  --config configs/experiments/sbm_report_graphlet_history.yaml
```

Train DegreeVAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml \
  --dataset sbm
```

Build the teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml
```

Train the learned selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/sbm_report_graphlet_history.yaml
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
  --config configs/experiments/sbm_report.yaml \
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
  --config configs/experiments/qm9_topology_mixture_catflow.yaml
```

Train topology DegreeVAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml \
  --dataset qm9_topology
```

Build topology teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml
```

Train topology selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/qm9_topology_mixture_catflow.yaml
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
  --config configs/experiments/qm9_topology_graphlet_history.yaml
```

Train topology DegreeVAE:

```bash
PYTHONPATH=src python scripts/train_degree_generator.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml \
  --dataset qm9_topology
```

Build topology teacher cache:

```bash
PYTHONPATH=src python scripts/build_rewiring_teacher.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml
```

Train topology selector:

```bash
PYTHONPATH=src python scripts/train_rewiring_selector.py \
  --config configs/experiments/qm9_topology_graphlet_history.yaml
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
  --config configs/experiments/qm9_topology_mixture_catflow.yaml
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
configs/experiments/sbm_report.yaml
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
  training_metrics.json
```

Summary generator outputs:

```text
outputs/summary_generators/<run_name>/
  checkpoint.pt
  summary_vectorizer.json
  training_metrics.json

outputs/target_summary_generators/<run_name>/
  checkpoint.pt
  summary_vectorizer.json
  training_metrics.json
  verification.json
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
1. Follow Part A, Steps 0--8.
2. Run the six component-isolating ablations in Section 6.
3. Compare against the legacy unconditional-summary pipeline.
4. Repeat the complete evaluation for seeds 42, 43, and 44.
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
