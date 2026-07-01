#!/bin/bash
set -u

# NT downstream evaluation launcher (18 tasks, 10-fold CV by default).
# Run from repo root: bash evaluation/run_eval.sh
# Edit GPUS to match your machine before launching.
# Format: task|learning_rate|effective_batch_size|physical_batch_size

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODEL="110m"  # "110m" or "2m"
if [ "$MODEL" = "110m" ]; then
  CHECKPOINT="models_ckpts/model_ckpt_110m.pt"
  OUT="results/nt_110m_ldarnet"
elif [ "$MODEL" = "2m" ]; then
  CHECKPOINT="models_ckpts/model_ckpt_2m.pt"
  OUT="results/nt_2m_ldarnet"
else
  echo "ERROR: MODEL must be '110m' or '2m'"; exit 1
fi

SCRIPT="$SCRIPT_DIR/eval_nt.py"
GPUS=(0 1 2)
BATCH=18   # max concurrent jobs (one per task)
FOLDS=10   # full NT protocol; use 3 for a quick smoke test
EPOCHS=100

TASKS=(
  "H3|1e-4|64|64"
  "H4|5e-5|64|64"
  "H3K9ac|1e-4|64|64"
  "H3K14ac|1e-4|64|64"
  "H4ac|2e-4|64|64"
  "H3K4me1|2e-4|64|64"
  "H3K4me2|2e-4|64|64"
  "H3K4me3|2e-4|64|64"
  "H3K36me3|1e-4|64|64"
  "H3K79me3|1e-4|64|64"
  "promoter_all|1e-4|64|64"
  "promoter_no_tata|2e-4|64|64"
  "promoter_tata|1e-4|64|64"
  "enhancers|2e-5|64|64"
  "enhancers_types|5e-5|64|64"
  "splice_sites_all|2e-4|64|64"
  "splice_sites_acceptors|2e-4|64|64"
  "splice_sites_donors|2e-4|64|64"
)

echo "MODEL=$MODEL  CHECKPOINT=$CHECKPOINT"
echo "OUT=$OUT  FOLDS=$FOLDS  EPOCHS=$EPOCHS  GPUS=${GPUS[*]}  BATCH=$BATCH  tasks=${#TASKS[@]}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"; exit 1
fi

mkdir -p "$OUT"

ngpu=${#GPUS[@]}
ntask=${#TASKS[@]}

running=0
i=0
while [ "$i" -lt "$ntask" ]; do
  IFS='|' read -r task lr eff phys <<< "${TASKS[$i]}"
  gpu=${GPUS[$(( i % ngpu ))]}
  echo "[launch $((i+1))/$ntask] $task  lr=$lr eff=$eff phys=$phys  -> GPU $gpu  (running: $((running+1)))"

  CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUT" \
    --subset_name "$task" \
    --learning_rate "$lr" \
    --effective_batch_size "$eff" \
    --physical_batch_size "$phys" \
    --num_folds "$FOLDS" \
    --epochs "$EPOCHS" \
    > "$OUT/${task}.log" 2>&1 &

  running=$(( running + 1 ))
  i=$(( i + 1 ))

  if [ "$running" -ge "$BATCH" ]; then
    wait -n || echo "WARN: a job exited with a non-zero status"
    running=$(( running - 1 ))
  fi
done

echo "Waiting for remaining jobs..."
wait
echo "Done. ${ntask} tasks finished. Summaries: $OUT/<task>/lr*_bs*/results_summary.json"
echo "Aggregate table: python evaluation/aggregate_results.py $OUT --markdown"
