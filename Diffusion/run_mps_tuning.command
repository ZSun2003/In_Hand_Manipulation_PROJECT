#!/bin/zsh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
RUNNER="$SCRIPT_DIR/run_screw.command"
TUNE_ROOT="$SCRIPT_DIR/mps_tune"
DATASET="$TUNE_ROOT/datasets/demos"
CHECKPOINTS="$TUNE_ROOT/checkpoints"
RUNS="$TUNE_ROOT/runs"
OUTPUTS="$TUNE_ROOT/outputs"
DIFFUSION="$CHECKPOINTS/screw_diffusion.pt"
RESIDUAL_STEM="$CHECKPOINTS/screw_residual_tqc"
RESIDUAL="$RESIDUAL_STEM.zip"
VECNORMALIZE="$RESIDUAL_STEM.vecnormalize.pkl"

# These defaults exercise every boundary with a deliberately small budget.
# Override them without editing this file when comparing hyperparameters.
DEMO_EPISODES="${SCREW_MPS_DEMO_EPISODES:-8}"
DIFFUSION_EPOCHS="${SCREW_MPS_DIFFUSION_EPOCHS:-5}"
WRENCH_STEPS="${SCREW_MPS_WRENCH_STEPS:-8000}"
NONE_STEPS="${SCREW_MPS_NONE_STEPS:-8000}"
EVAL_EPISODES="${SCREW_MPS_EVAL_EPISODES:-3}"

mkdir -p "$DATASET" "$CHECKPOINTS" "$RUNS" "$OUTPUTS"

usage() {
  print "Usage: ./run_mps_tuning.command <stage> [extra arguments]"
  print ""
  print "Stages:"
  print "  demos       Generate a small isolated expert dataset"
  print "  diffusion   Train the isolated Diffusion checkpoint on MPS"
  print "  wrench      Train the first small residual TQC stage"
  print "  none        Resume residual TQC with physical guide disabled"
  print "  eval        Evaluate the isolated final hybrid stack"
  print "  gui         View the best available isolated policy"
  print "  all         Run demos -> diffusion -> wrench -> none -> eval"
  print "  paths       Print the isolated tuning artifact paths"
  print ""
  print "Defaults are pipeline checks, not convergence budgets:"
  print "  demos=$DEMO_EPISODES, diffusion_epochs=$DIFFUSION_EPOCHS,"
  print "  wrench_steps=$WRENCH_STEPS, none_steps=$NONE_STEPS,"
  print "  eval_episodes=$EVAL_EPISODES"
  print ""
  print "Extra arguments are appended after stage defaults, so argparse uses"
  print "the final repeated value for options such as --epochs or --timesteps."
}

require_file() {
  if [[ ! -f "$1" ]]; then
    print -u2 "Required tuning artifact is missing: $1"
    print -u2 "Run the preceding stage first."
    exit 2
  fi
}

stage_demos() {
  "$RUNNER" generate_demos.py \
    --episodes "$DEMO_EPISODES" \
    --output "$DATASET" \
    --obs-horizon 2 \
    --action-horizon 16 \
    --overwrite \
    "$@"
}

stage_diffusion() {
  require_file "$DATASET/metadata.json"
  "$RUNNER" train_diffusion.py \
    --dataset "$DATASET" \
    --output "$DIFFUSION" \
    --epochs "$DIFFUSION_EPOCHS" \
    --batch-size 256 \
    --cache-size "$DEMO_EPISODES" \
    --device mps \
    "$@"
}

stage_wrench() {
  require_file "$DIFFUSION"
  "$RUNNER" train_residual.py \
    --baseline diffusion \
    --diffusion-checkpoint "$DIFFUSION" \
    --diffusion-device mps \
    --guide-mode wrench \
    --guide-scale 0.35 \
    --timesteps "$WRENCH_STEPS" \
    --workers 2 \
    --batch-size 128 \
    --buffer-size 25000 \
    --learning-starts 512 \
    --train-freq 1 \
    --gradient-steps 1 \
    --n-critics 2 \
    --n-quantiles 25 \
    --top-quantiles-to-drop 2 \
    --device mps \
    --tensorboard-dir "$RUNS" \
    --output "$RESIDUAL_STEM" \
    "$@"
}

stage_none() {
  require_file "$DIFFUSION"
  require_file "$RESIDUAL"
  require_file "$VECNORMALIZE"
  "$RUNNER" train_residual.py \
    --baseline diffusion \
    --diffusion-checkpoint "$DIFFUSION" \
    --diffusion-device mps \
    --guide-mode none \
    --guide-scale 0 \
    --timesteps "$NONE_STEPS" \
    --workers 2 \
    --batch-size 128 \
    --buffer-size 25000 \
    --learning-starts 512 \
    --train-freq 1 \
    --gradient-steps 1 \
    --n-critics 2 \
    --n-quantiles 25 \
    --top-quantiles-to-drop 2 \
    --device mps \
    --tensorboard-dir "$RUNS" \
    --resume "$RESIDUAL" \
    --output "$RESIDUAL_STEM" \
    "$@"
}

stage_eval() {
  require_file "$DIFFUSION"
  require_file "$RESIDUAL"
  require_file "$VECNORMALIZE"
  "$RUNNER" evaluate.py \
    --mode hybrid \
    --episodes "$EVAL_EPISODES" \
    --guide-mode none \
    --guide-scale 0 \
    --diffusion-checkpoint "$DIFFUSION" \
    --residual-checkpoint "$RESIDUAL" \
    --vecnormalize "$VECNORMALIZE" \
    --output "$OUTPUTS/eval_hybrid_none.json" \
    "$@"
}

stage_gui() {
  if [[ -f "$RESIDUAL" && -f "$VECNORMALIZE" && -f "$DIFFUSION" ]]; then
    "$RUNNER" run_screw.py \
      --mode hybrid \
      --guide-mode none \
      --diffusion-checkpoint "$DIFFUSION" \
      --residual-checkpoint "$RESIDUAL" \
      --vecnormalize "$VECNORMALIZE" \
      "$@"
  elif [[ -f "$DIFFUSION" ]]; then
    print "Residual tuning checkpoint not found; opening Diffusion only."
    "$RUNNER" run_screw.py \
      --mode diffusion \
      --guide-mode none \
      --diffusion-checkpoint "$DIFFUSION" \
      "$@"
  else
    print "No tuning checkpoint found; opening the expert demonstration."
    "$RUNNER" run_screw.py --mode expert "$@"
  fi
}

stage_paths() {
  print "MPS tuning root: $TUNE_ROOT"
  print "Dataset:         $DATASET"
  print "Diffusion:       $DIFFUSION"
  print "Residual:        $RESIDUAL"
  print "VecNormalize:    $VECNORMALIZE"
  print "TensorBoard:     $RUNS"
  print "Evaluation:      $OUTPUTS/eval_hybrid_none.json"
}

stage="${1:-help}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

case "$stage" in
  demos)
    stage_demos "$@"
    ;;
  diffusion)
    stage_diffusion "$@"
    ;;
  wrench)
    stage_wrench "$@"
    ;;
  none)
    stage_none "$@"
    ;;
  eval)
    stage_eval "$@"
    ;;
  gui)
    stage_gui "$@"
    ;;
  all)
    if [[ "$#" -ne 0 ]]; then
      print -u2 "'all' does not accept stage-specific extra arguments."
      print -u2 "Use environment budget variables or run stages separately."
      exit 2
    fi
    stage_demos
    stage_diffusion
    stage_wrench
    stage_none
    stage_eval
    print ""
    print "MPS tuning pipeline completed. Open it with:"
    print "  ./run_mps_tuning.command gui"
    ;;
  paths)
    stage_paths
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    print -u2 "Unknown stage: $stage"
    usage >&2
    exit 2
    ;;
esac
