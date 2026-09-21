# GraphER Tables 6 and 7 ablation commands

These commands use only the existing `scripts/run_gdsm_simple_baseline.py` and
`scripts/evaluate_graph_generation_report.py`. No new runner/evaluator script is required.

The main trained run/checkpoint is reused for every sampling ablation.
Do **not** rerun training.

## Table 6: rewiring ablation

Variants:
- `no_rewiring`: same checkpoint and learned degree prior; `guidance.enabled=false`.
- `unguided_rewiring`: same checkpoint/prior/move budget; uniformly selects among admissible same-type swaps and does not use learned energy for selection.
- `guided`: existing main GraphER generation. Reuse `seed_${SEED}_n_1024` or regenerate with the original `gdsm_final_explicit` config.

## Table 7: degree source ablation

Variants:
- `learned_prior`: existing main GraphER generation (`seed_${SEED}_n_1024`).
- `empirical_degree`: same trained checkpoint and guided refiner; the spectral-anchor degree source is sampled from training degree sequences.

---

# COMMUNITY-SMALL

## Seed 42 - no rewiring

```bash
RUN=seed_42_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_42_no_rewiring.yaml
GEN_ID=seed_42_n_1024_no_rewiring

PYTHONHASHSEED=42 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 42 \
  --run-id "$RUN" \
  --generation-seed 42 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_no_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 42 - unguided rewiring

```bash
RUN=seed_42_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_42_unguided_rewiring.yaml
GEN_ID=seed_42_n_1024_unguided_rewiring

PYTHONHASHSEED=42 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 42 \
  --run-id "$RUN" \
  --generation-seed 42 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_unguided_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 42 - empirical degree source

```bash
RUN=seed_42_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_42_empirical_degree.yaml
GEN_ID=seed_42_n_1024_empirical_degree

PYTHONHASHSEED=42 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 42 \
  --run-id "$RUN" \
  --generation-seed 42 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_empirical_degree \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```


# COMMUNITY_SMALL - SEED 43

## Seed 43 - no rewiring

```bash
RUN=seed_43_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_43_no_rewiring.yaml
GEN_ID=seed_43_n_1024_no_rewiring

PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id "$RUN" \
  --generation-seed 43 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=43 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_no_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 43 - unguided rewiring

```bash
RUN=seed_43_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_43_unguided_rewiring.yaml
GEN_ID=seed_43_n_1024_unguided_rewiring

PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id "$RUN" \
  --generation-seed 43 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=43 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_unguided_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 43 - empirical degree

```bash
RUN=seed_43_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_43_empirical_degree.yaml
GEN_ID=seed_43_n_1024_empirical_degree

PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id "$RUN" \
  --generation-seed 43 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=43 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_empirical_degree \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

# COMMUNITY_SMALL - SEED 44

## Seed 44 - no rewiring

```bash
RUN=seed_44_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_44_no_rewiring.yaml
GEN_ID=seed_44_n_1024_no_rewiring

PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id "$RUN" \
  --generation-seed 44 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=44 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_no_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 44 - unguided rewiring

```bash
RUN=seed_44_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_44_unguided_rewiring.yaml
GEN_ID=seed_44_n_1024_unguided_rewiring

PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id "$RUN" \
  --generation-seed 44 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=44 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_unguided_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 44 - empirical degree

```bash
RUN=seed_44_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/community_small_seed_44_empirical_degree.yaml
GEN_ID=seed_44_n_1024_empirical_degree

PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset community_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset sbm \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id "$RUN" \
  --generation-seed 44 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/community_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=44 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/community_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_empirical_degree \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

# EGO_SMALL - SEED 42

## Seed 42 - no rewiring

```bash
RUN=seed_42_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_42_no_rewiring.yaml
GEN_ID=seed_42_n_1024_no_rewiring

PYTHONHASHSEED=42 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 42 \
  --run-id "$RUN" \
  --generation-seed 42 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_no_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 42 - unguided rewiring

```bash
RUN=seed_42_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_42_unguided_rewiring.yaml
GEN_ID=seed_42_n_1024_unguided_rewiring

PYTHONHASHSEED=42 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 42 \
  --run-id "$RUN" \
  --generation-seed 42 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_unguided_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 42 - empirical degree

```bash
RUN=seed_42_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_42_empirical_degree.yaml
GEN_ID=seed_42_n_1024_empirical_degree

PYTHONHASHSEED=42 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 42 \
  --run-id "$RUN" \
  --generation-seed 42 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=42 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_empirical_degree \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

# EGO_SMALL - SEED 43

## Seed 43 - no rewiring

```bash
RUN=seed_43_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_43_no_rewiring.yaml
GEN_ID=seed_43_n_1024_no_rewiring

PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id "$RUN" \
  --generation-seed 43 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=43 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_no_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 43 - unguided rewiring

```bash
RUN=seed_43_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_43_unguided_rewiring.yaml
GEN_ID=seed_43_n_1024_unguided_rewiring

PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id "$RUN" \
  --generation-seed 43 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=43 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_unguided_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 43 - empirical degree

```bash
RUN=seed_43_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_43_empirical_degree.yaml
GEN_ID=seed_43_n_1024_empirical_degree

PYTHONHASHSEED=43 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 43 \
  --run-id "$RUN" \
  --generation-seed 43 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=43 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_empirical_degree \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

# EGO_SMALL - SEED 44

## Seed 44 - no rewiring

```bash
RUN=seed_44_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_44_no_rewiring.yaml
GEN_ID=seed_44_n_1024_no_rewiring

PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id "$RUN" \
  --generation-seed 44 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=44 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_no_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 44 - unguided rewiring

```bash
RUN=seed_44_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_44_unguided_rewiring.yaml
GEN_ID=seed_44_n_1024_unguided_rewiring

PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id "$RUN" \
  --generation-seed 44 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=44 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_unguided_rewiring \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

## Seed 44 - empirical degree

```bash
RUN=seed_44_gdsm_final_g345
N=1024
CFG=configs/experiments/gdsm_final_ablation/ego_small_seed_44_empirical_degree.yaml
GEN_ID=seed_44_n_1024_empirical_degree

PYTHONHASHSEED=44 GDSM_EIGH_BACKEND=cpu PYTHONPATH=src \
python scripts/run_gdsm_simple_baseline.py \
  --stage generate \
  --dataset ego_small \
  --no-common-config \
  --wrapper-config "$CFG" \
  --dataset-root outputs/datasets \
  --serialized-dataset ego_small \
  --output-root outputs/baselines \
  --seed-id 44 \
  --run-id "$RUN" \
  --generation-seed 44 \
  --generation-id "$GEN_ID" \
  --num-samples "$N" \
  --device cuda:0

GEN="outputs/baselines/gdsm_simple/ego_small/$RUN/generations/$GEN_ID"
PYTHONHASHSEED=44 ORCA_EXEC=/home/quang/orca/orca.out PYTHONPATH=src \
python scripts/evaluate_graph_generation_report.py \
  --config configs/experiments/baselines/ego_small_evaluation.yaml \
  --generated-dir "$GEN" \
  --generated-graphs "$GEN/base_graphs.pkl" \
  --base-graphs "$GEN/initial_graphs.pkl" \
  --generated-stage gdsm_final_empirical_degree \
  --reference-split test \
  --generic-mmd-protocol graphrnn \
  --num-samples "$N" \
  --output-dir "$GEN/evaluation_topology"
```

# Existing guided / learned-prior rows

For Table 6 `GraphER rewiring` and Table 7 `Learned prior`, reuse the existing main outputs:

- `outputs/baselines/gdsm_simple/community_small/seed_42_gdsm_final_g345/generations/seed_42_n_1024`
- corresponding seed 43 / 44 directories
- same layout under `ego_small`

If you want to regenerate them, use the original `configs/experiments/gdsm_final_explicit/<dataset>_seed_<seed>.yaml` with the same run id and generation seed, but a distinct generation id such as `seed_<seed>_n_1024_guided_rerun`.
