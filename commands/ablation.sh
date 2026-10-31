#!/usr/bin/env bash
# Paper Section 5.2 / Appendix F.6-F.7. Run from any directory.
# Examples:
#   bash commands/ablation.sh community_small
#   bash commands/ablation.sh generic        # Community-small + Ego-small
#   bash commands/ablation.sh molecular      # QM9 + ZINC paired T/TR
#   DO_TRAIN=1 bash commands/ablation.sh all
#   RUN_S0=1 bash commands/ablation.sh generic  # separately retrains S0
#   RUN_S0=1 DO_GENERATE=0 DO_S0_GENERATE=1 bash commands/ablation.sh generic
#   DRY_RUN=1 SEEDS='42' bash commands/ablation.sh all
# Existing checkpoints are reused by default. All model/sampling parameters
# remain explicit in configs/experiments/grapher_paper_ablation/*.yaml.
# Dataset/study entry points live in commands/ablations/<dataset>/ and set
# REUSE=1: validate and reuse one population, or generate it only when absent.
# Direct invocation retains the original all-studies interface (REUSE=0).
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-community_small}"
case "$TARGET" in
  community_small|ego_small|qm9|zinc) DATASETS=("$TARGET") ;;
  generic) DATASETS=(community_small ego_small) ;;
  molecular) DATASETS=(qm9 zinc) ;;
  all) DATASETS=(community_small ego_small qm9 zinc) ;;
  *) echo "Usage: bash commands/ablation.sh {community_small|ego_small|qm9|zinc|generic|molecular|all}" >&2; exit 2 ;;
esac
ABLATION_STUDY="${ABLATION_STUDY:-all}"
case "$ABLATION_STUDY" in
  all|primary|decoder_temperature|rewiring_budget|summary_supervision|rewiring) ;;
  *) echo "Unknown ABLATION_STUDY: $ABLATION_STUDY" >&2; exit 2 ;;
esac
for dataset in "${DATASETS[@]}"; do
  case "$ABLATION_STUDY:$dataset" in
    all:*|rewiring:qm9|rewiring:zinc|primary:community_small|primary:ego_small|decoder_temperature:community_small|decoder_temperature:ego_small|rewiring_budget:community_small|rewiring_budget:ego_small|summary_supervision:community_small|summary_supervision:ego_small) ;;
    *) echo "Study $ABLATION_STUDY is not implemented for $dataset" >&2; exit 2 ;;
  esac
done

DEVICE="${DEVICE:-cuda:0}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-test}"
DATASET_ROOT="${DATASET_ROOT:-outputs/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baselines}"
REPORT_ROOT="${REPORT_ROOT:-outputs/ablations/grapher_paper}"
ORCA_EXEC="${ORCA_EXEC:-/home/quang/orca/orca.out}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DO_TRAIN="${DO_TRAIN:-0}"
DO_GENERATE="${DO_GENERATE:-1}"
DO_EVALUATE="${DO_EVALUATE:-1}"
RUN_S0="${RUN_S0:-0}"
if [[ "$ABLATION_STUDY" == summary_supervision ]]; then
  RUN_S0=1
elif [[ "$ABLATION_STUDY" != all ]]; then
  RUN_S0=0
fi
DO_S0_GENERATE="${DO_S0_GENERATE:-$DO_GENERATE}"
DO_S0_TRAIN="${DO_S0_TRAIN:-$DO_S0_GENERATE}"
REUSE="${REUSE:-0}"
GENERATION_TAG="${GENERATION_TAG:-paper_ablation_v1}"
S0_GENERATION_TAG="${S0_GENERATION_TAG:-paper_S0_v1}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a SEED_VALUES <<< "${SEEDS:-41 42 43 44 45}"
[[ "$REFERENCE_SPLIT" == test || "$REFERENCE_SPLIT" == val ]] || { echo "REFERENCE_SPLIT must be test or val" >&2; exit 2; }
for value in "$DO_TRAIN" "$DO_GENERATE" "$DO_EVALUATE" "$RUN_S0" "$DO_S0_GENERATE" "$DO_S0_TRAIN" "$REUSE" "$OVERWRITE" "$DRY_RUN"; do
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "Boolean switches must be 0 or 1" >&2; exit 2; }
done
for tag in "$GENERATION_TAG" "$S0_GENERATION_TAG"; do
  [[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Generation tags must be safe single path components" >&2; exit 2; }
done
[[ ${#SEED_VALUES[@]} -gt 0 ]] || { echo "At least one training seed is required" >&2; exit 2; }
for seed in "${SEED_VALUES[@]}"; do
  [[ "$seed" =~ ^(41|42|43|44|45)$ ]] || { echo "Provided configs cover training seeds 41--45" >&2; exit 2; }
done
OVERWRITE_ARGS=()
[[ "$OVERWRITE" == 0 ]] || OVERWRITE_ARGS=(--overwrite)
SUMMARY_ARGS=()
[[ "$DO_EVALUATE" == 0 ]] || SUMMARY_ARGS=(--require-complete)

run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

train_or_generate() {
  local stage="$1" config="$2" run_id="$3" gen_id="$4"
  local extra=()
  if [[ "$stage" == generate ]]; then
    extra=(--generation-seed "$SEED" --generation-id "$gen_id" --num-samples "$NUM_REQUESTED")
  fi
  run env PYTHONHASHSEED="$SEED" GRAPHER_EIGH_BACKEND=cpu PYTHONPATH=src \
    "$PYTHON_BIN" scripts/run_grapher_baseline.py \
    --stage "$stage" --dataset "$DATASET" --no-common-config \
    --wrapper-config "$config" --dataset-root "$DATASET_ROOT" \
    --serialized-dataset "$SERIALIZED" --output-root "$OUTPUT_ROOT" \
    --seed-id "$SEED" --run-id "$run_id" --device "$DEVICE" \
    "${extra[@]}" "${OVERWRITE_ARGS[@]}"
}

check_artifacts() {
  local mode="$1" config="$2" run_id="$3" gen_id="$4" kind="$5"
  shift 5
  local run_dir="$OUTPUT_ROOT/grapher/$DATASET/$run_id"
  local extra=()
  if [[ "$mode" != training ]]; then
    extra=(--generation-dir "$run_dir/generations/$gen_id" --num-samples "$NUM_REQUESTED" --kind "$kind")
  fi
  run env PYTHONPATH=src "$PYTHON_BIN" scripts/check_grapher_ablation_artifacts.py \
    "$mode" --run-dir "$run_dir" --config "$config" --dataset "$DATASET" \
    --serialized-dataset "$SERIALIZED" --dataset-root "$DATASET_ROOT" --seed "$SEED" \
    "${extra[@]}" "$@"
}

with_artifact_lock() {
  local lock_path="$1"; shift
  if [[ "$DRY_RUN" == 1 ]]; then "$@"; return $?; fi
  command -v flock >/dev/null || { echo "Safe concurrent reuse requires the Linux flock utility" >&2; return 2; }
  mkdir -p "$(dirname "$lock_path")"
  ( flock -x 9 || exit $?; "$@" ) 9>"$lock_path"
}

ensure_training_unlocked() {
  local config="$1" run_id="$2" gen_id="$3" kind="$4" allow_train="$5"
  local status=0
  if [[ "$DRY_RUN" == 1 ]]; then
    check_artifacts training "$config" "$run_id" "$gen_id" "$kind"
    [[ "$allow_train" == 0 ]] || train_or_generate train "$config" "$run_id" "$gen_id"
    return 0
  fi
  if check_artifacts training "$config" "$run_id" "$gen_id" "$kind"; then return 0; else status=$?; fi
  [[ "$status" == 3 ]] || return "$status"
  [[ "$allow_train" == 1 ]] || { echo "Checkpoint absent; enable the corresponding training switch to create it" >&2; return 3; }
  train_or_generate train "$config" "$run_id" "$gen_id" || return $?
  check_artifacts training "$config" "$run_id" "$gen_id" "$kind"
}

ensure_generation_unlocked() {
  local config="$1" train_config="$2" run_id="$3" gen_id="$4" kind="$5" allow_train="$6" allow_generate="$7"
  local status=0 run_dir="$OUTPUT_ROOT/grapher/$DATASET/$run_id"
  if [[ "$DRY_RUN" == 1 ]]; then
    # Show both the validation and the conditional creation plan; do not probe
    # real output paths or infer that a cache exists during a dry run.
    check_artifacts generation "$config" "$run_id" "$gen_id" "$kind"
    with_artifact_lock "$run_dir/.ablation_training.lock" ensure_training_unlocked "$train_config" "$run_id" "$gen_id" "$kind" "$allow_train"
    if [[ "$allow_generate" == 1 ]]; then
      check_artifacts begin "$config" "$run_id" "$gen_id" "$kind"
      train_or_generate generate "$config" "$run_id" "$gen_id"
      check_artifacts seal "$config" "$run_id" "$gen_id" "$kind"
    fi
    return 0
  fi
  if check_artifacts generation "$config" "$run_id" "$gen_id" "$kind"; then
    echo "Reusing verified population: $run_dir/generations/$gen_id"
    return 0
  else status=$?; fi
  [[ "$status" == 3 ]] || return "$status"
  with_artifact_lock "$run_dir/.ablation_training.lock" ensure_training_unlocked "$train_config" "$run_id" "$gen_id" "$kind" "$allow_train" || return $?
  [[ "$allow_generate" == 1 ]] || { echo "Population absent and generation is disabled: $gen_id" >&2; return 3; }
  check_artifacts begin "$config" "$run_id" "$gen_id" "$kind" || return $?
  train_or_generate generate "$config" "$run_id" "$gen_id" || return $?
  check_artifacts seal "$config" "$run_id" "$gen_id" "$kind"
}

ensure_generation() {
  local run_id="$3" gen_id="$4"
  with_artifact_lock "$OUTPUT_ROOT/grapher/$DATASET/$run_id/generations/.$gen_id.ablation.lock" ensure_generation_unlocked "$@"
}

evaluate_branch() {
  local branch="$1" branch_dir="$GEN/ablations/$1"
  if [[ "$DRY_RUN" != 1 && ! -f "$branch_dir/base_graphs.pkl" ]]; then
    echo "Missing branch $branch for seed $SEED; the summary will record it." >&2
    return 1
  fi
  local evaluation="$branch_dir/$EVALUATION_SUBDIR"
  if [[ "$IS_GENERIC" == 1 ]]; then
    run env PYTHONHASHSEED="$SEED" ORCA_EXEC="$ORCA_EXEC" PYTHONPATH=src \
      "$PYTHON_BIN" scripts/evaluate_graph_generation_report.py \
      --config "configs/evaluation/${DATASET}.yaml" \
      --dataset-root "$DATASET_ROOT" \
      --reference-split "$REFERENCE_SPLIT" --generic-mmd-protocol graphrnn \
      --generic-validity-require-connected --generic-validity-min-nodes 1 \
      --generic-validity-max-nodes "$MAX_NODES" \
      --generated-dir "$branch_dir" --generated-graphs "$branch_dir/base_graphs.pkl" \
      --generated-stage "$branch" --output-dir "$evaluation" || return $?
  else
    local molecular_extra=()
    [[ "$DATASET" != zinc ]] || molecular_extra=(--graphlet-num-samples 4096)
    run env PYTHONHASHSEED="$SEED" PYTHONPATH=src \
      "$PYTHON_BIN" scripts/evaluate_generated_molecules.py \
      --generated-dir "$GEN" --generated-graphs "$branch_dir/base_graphs.pkl" \
      --dataset-root "$DATASET_ROOT" --dataset "$SERIALIZED" \
      --reference-split "$REFERENCE_SPLIT" --train-split train \
      --output-dir "$evaluation" --strict-raw-metrics --metric-molecule-source raw_valid \
      --nspdk-backend eden --nspdk-complexity 4 --nspdk-bond-label-mode hogdiff \
      --require-fcd --fcd-device "$DEVICE" --graphlet-mmd \
      --graphlet-k-min 3 --graphlet-k-max 5 --graphlet-topology-filter all \
      --graphlet-node-attribute atomic_num --graphlet-edge-attribute bond_type \
      --graphlet-attributed-backend python "${molecular_extra[@]}" || return $?
    run env PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_qed.py \
      --generated-graphs "$branch_dir/base_graphs.pkl" \
      --reference-graphs "$DATASET_ROOT/$SERIALIZED/${REFERENCE_SPLIT}.pkl" \
      --molecular-report "$evaluation/molecular_evaluation_metrics.json" \
      --expected-num-generated "$NUM_REQUESTED" --output-dir "$evaluation/qed" || return $?
  fi
}

prepare_seed() {
  local base_config="configs/experiments/grapher_final/${DATASET}_seed_${SEED}.yaml"
  local ablation_config="configs/experiments/grapher_paper_ablation/${DATASET}_seed_${SEED}_${CONFIG_SUFFIX}.yaml"
  local kind=matched
  [[ "$IS_GENERIC" == 1 ]] || kind=molecular
  if [[ "$REUSE" == 1 ]]; then
    ensure_generation "$ablation_config" "$base_config" "$RUN" "$GEN_ID" "$kind" "$DO_TRAIN" "$DO_GENERATE" || return $?
  else
    if [[ "$DO_TRAIN" == 1 ]]; then train_or_generate train "$base_config" "$RUN" "$GEN_ID" || return $?; fi
    if [[ "$DO_GENERATE" == 1 ]]; then train_or_generate generate "$ablation_config" "$RUN" "$GEN_ID" || return $?; fi
  fi
  if [[ "$IS_GENERIC" == 0 ]]; then
    local export_args=("${OVERWRITE_ARGS[@]}")
    [[ "$REUSE" == 0 ]] || export_args=()
    run env PYTHONPATH=src "$PYTHON_BIN" scripts/export_grapher_ablation_views.py molecular \
      --generation-dir "$GEN" "${export_args[@]}" || return $?
  fi
  if [[ "$REUSE" == 1 ]]; then
    local primary_variants=()
    for variant in "${VARIANTS[@]}"; do [[ "$variant" == S0 ]] || primary_variants+=("$variant"); done
    check_artifacts branches "$ablation_config" "$RUN" "$GEN_ID" "$kind" --variants "${primary_variants[@]}" || return $?
  fi
}

prepare_s0() {
    local s0_config="configs/experiments/grapher_paper_ablation/${DATASET}_seed_${SEED}_s0.yaml"
    local s0_run="${RUN}_S0" s0_id="${S0_GENERATION_TAG}_seed_${SEED}_n_${NUM_REQUESTED}"
    local s0_gen="$OUTPUT_ROOT/grapher/$DATASET/$s0_run/generations/$s0_id"
    if [[ "$REUSE" == 1 ]]; then
      ensure_generation "$s0_config" "$s0_config" "$s0_run" "$s0_id" s0 "$DO_S0_TRAIN" "$DO_S0_GENERATE" || return $?
    else
      if [[ "$DO_S0_TRAIN" == 1 ]]; then train_or_generate train "$s0_config" "$s0_run" "$s0_id" || return $?; fi
      if [[ "$DO_S0_GENERATE" == 1 ]]; then train_or_generate generate "$s0_config" "$s0_run" "$s0_id" || return $?; fi
    fi
    local export_args=("${OVERWRITE_ARGS[@]}")
    [[ "$REUSE" == 0 ]] || export_args=()
    run env PYTHONPATH=src "$PYTHON_BIN" scripts/export_grapher_ablation_views.py s0 \
      --generation-dir "$s0_gen" --output-root "$GEN/ablations" \
      --reference-generation-dir "$GEN" "${export_args[@]}" || return $?
}

evaluation_status() {
  [[ "$DRY_RUN" != 1 ]] || return 0
  # A failed atomic generation may have no output directory. Do not create one:
  # it would block the next generation's existing-artifact guard.
  [[ -d "$GEN" ]] || return 0
  local destination="$GEN/ablations/$1/$EVALUATION_SUBDIR"
  mkdir -p "$destination"
  # The summarizer honors this marker before reading any previous report.
  printf '{"status": "%s", "exit_code": %d}\n' "$2" "$3" > "$destination/evaluation_status.json"
}

FAILED=0
for DATASET in "${DATASETS[@]}"; do
  case "$DATASET" in
    community_small) SERIALIZED=sbm; MAX_NODES=20; IS_GENERIC=1 ;;
    ego_small) SERIALIZED=ego_small; MAX_NODES=18; IS_GENERIC=1 ;;
    qm9) SERIALIZED=qm9_attributed; MAX_NODES=9; IS_GENERIC=0 ;;
    zinc) SERIALIZED=zinc; MAX_NODES=38; IS_GENERIC=0 ;;
  esac
  if [[ "$IS_GENERIC" == 1 ]]; then
    NUM_REQUESTED="${N:-1024}"; PROFILE=tau0p4_B1to4; CONFIG_SUFFIX=matched
    VARIANTS=(T D TR DR D0 R1 R12)
    [[ "$RUN_S0" == 0 ]] || VARIANTS+=(S0)
  else
    NUM_REQUESTED="${N:-10000}"; PROFILE=B1to4; CONFIG_SUFFIX=paired
    VARIANTS=(T TR)
  fi
  case "$ABLATION_STUDY" in
    primary) VARIANTS=(T D TR DR) ;;
    decoder_temperature) VARIANTS=(D D0) ;;
    rewiring_budget) VARIANTS=(D R1 R12 DR) ;;
    summary_supervision) VARIANTS=(D S0) ;;
    rewiring) VARIANTS=(T TR) ;;
  esac
  EVALUATION_SUBDIR="evaluation_${REFERENCE_SPLIT}"
  STUDY_REPORT_ROOT="$REPORT_ROOT/$DATASET"
  if [[ "$ABLATION_STUDY" != all ]]; then
    EVALUATION_SUBDIR+="/$ABLATION_STUDY"
    STUDY_REPORT_ROOT+="/$ABLATION_STUDY"
  fi
  [[ "$NUM_REQUESTED" =~ ^[1-9][0-9]*$ ]] || { echo "N must be positive" >&2; exit 2; }
  SEED_DIR_ARGS=()
  for SEED in "${SEED_VALUES[@]}"; do
    # Reuse the verified training-run names already present on the user's server.
    # Keep these distinct from GENERATION_TAG: the paper ablation creates new
    # generations from an existing checkpoint and never needs to retrain it.
    if [[ -n "${RUN_TEMPLATE:-}" ]]; then
      RUN="$RUN_TEMPLATE"
    else
      case "$DATASET" in
        community_small|ego_small|zinc)
          RUN='seed_{seed}_{dataset}_grapher_final_g345_v1' ;;
        qm9)
          RUN='seed_{seed}_{dataset}_grapher_final_B1to4_g345_v2' ;;
      esac
    fi
    RUN="${RUN//\{seed\}/$SEED}"
    RUN="${RUN//\{dataset\}/$DATASET}"
    RUN="${RUN//\{profile\}/$PROFILE}"
    GEN_ID="${GENERATION_TAG}_seed_${SEED}_n_${NUM_REQUESTED}"
    GEN="$OUTPUT_ROOT/grapher/$DATASET/$RUN/generations/$GEN_ID"
    SEED_DIR_ARGS+=(--seed-dir "$SEED=$GEN/ablations")
    echo "Paper ablation | dataset=$DATASET | study=$ABLATION_STUDY | seed=$SEED | requested=$NUM_REQUESTED"
    if prepare_seed; then
      if [[ "$DRY_RUN" != 1 ]]; then
        rm -f "$GEN.ablation_command_failure.json"
      fi
    else
      STATUS=$?
      echo "Seed $SEED failed for $DATASET; no replacement samples will be generated." >&2
      FAILED=1
      if [[ "$DRY_RUN" != 1 ]]; then
        mkdir -p "$(dirname "$GEN")"
        printf '{"status": "failed", "stage": "prepare_seed", "generation_seed": %d, "num_requested": %d, "exit_code": %d, "error": "prepare_seed command failed; see terminal log"}\n' \
          "$SEED" "$NUM_REQUESTED" "$STATUS" > "$GEN.ablation_command_failure.json"
      fi
      for BRANCH in "${VARIANTS[@]}"; do
        evaluation_status "$BRANCH" failed 1
      done
      continue
    fi
    S0_READY=1
    if [[ "$IS_GENERIC" == 1 && "$RUN_S0" == 1 ]]; then
      if ! prepare_s0; then
        echo "S0 failed for $DATASET seed=$SEED; primary branches remain available." >&2
        FAILED=1
        S0_READY=0
        evaluation_status S0 failed 1
      fi
    fi
    if [[ "$DO_EVALUATE" == 1 ]]; then
      for BRANCH in "${VARIANTS[@]}"; do
        [[ "$BRANCH" != S0 || "$S0_READY" == 1 ]] || continue
        evaluation_status "$BRANCH" running 0
        if evaluate_branch "$BRANCH"; then
          evaluation_status "$BRANCH" complete 0
        else
          STATUS=$?
          evaluation_status "$BRANCH" failed "$STATUS"
          echo "Evaluation failed: $DATASET seed=$SEED branch=$BRANCH" >&2
          FAILED=1
        fi
      done
    fi
  done
  if ! run "$PYTHON_BIN" scripts/summarize_grapher_ablation.py \
    "${SEED_DIR_ARGS[@]}" --variants "${VARIANTS[@]}" \
    "${SUMMARY_ARGS[@]}" \
    --evaluation-subdir "$EVALUATION_SUBDIR" \
    --output-dir "$STUDY_REPORT_ROOT/n_${NUM_REQUESTED}/evaluation_${REFERENCE_SPLIT}"; then
    FAILED=1
  fi
done
exit "$FAILED"
