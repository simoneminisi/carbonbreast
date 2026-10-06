#!/usr/bin/env bash
# Incremental Pareto step: add SGD (optimizer axis) + ViT (architecture axis) points.
# Head-only is pruned to resnet18 only, so every run here is full fine-tune.
# Each run's tag is checked against results.csv and SKIPPED if already present, so you can
# append more configs below and re-run safely to grow the frontier step by step.
#
#   bash run_next.sh
set -euo pipefail

IMG=${IMG:-768}; EPOCHS=${EPOCHS:-15}; ROUNDS=${ROUNDS:-1}; SEED=${SEED:-42}
RESULTS=${RESULTS:-results.csv}
# cap workers: huge --workers floods DataLoader IPC -> bus error (see run_pareto.sh)
WORKERS=${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}

# config rows:  backbone  use_trunk(0/1)  optimizer  batch  lr  extra_flags
CONFIGS=(
  "resnet18                1 sgd   16 0.01 "                 # optimizer axis on the frontier point
  "resnet18                0 sgd   16 0.01 "
  "convnext_small          1 sgd    8 0.01 "                 # optimizer on the strongest conv
  "vit_tiny_patch16_224    0 adamw  8 1e-4 --grad-checkpoint"  # ViT (no trunk, dynamic@768)
  "vit_small_patch16_224   0 adamw  4 1e-4 --grad-checkpoint"
)

for row in "${CONFIGS[@]}"; do
  read -r BB USE_TRUNK OPT BS LR EXTRA <<<"$row"
  tag="${BB}_${OPT}_full_q-fp16_img${IMG}_bs${BS}_ep${EPOCHS}_r${ROUNDS}_s${SEED}_trunk-${USE_TRUNK}"
  if [ -f "$RESULTS" ] && grep -q ",${tag}," "$RESULTS"; then
    echo "SKIP (already in $RESULTS): $tag"; continue
  fi
  TRUNK_ARG=()
  if [ "$USE_TRUNK" = 1 ]; then
    [ -f "trunk_${BB}.pt" ] || { echo "ERROR: trunk_${BB}.pt missing (run train_patch.py --backbone $BB first)"; exit 1; }
    TRUNK_ARG=(--pretrained-trunk "trunk_${BB}.pt")
  fi
  echo "===== RUN: $tag ====="
  python s_main.py --backbone "$BB" "${TRUNK_ARG[@]}" \
    --train-mode full --optimizer "$OPT" --lr "$LR" --quantize fp16 $EXTRA \
    --img-size "$IMG" --batch "$BS" --workers "$WORKERS" \
    --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" \
    --cache-ram --no-plot --results-csv "$RESULTS"
done
echo "DONE — regenerate the plot:  python ../common/plot_pareto.py $RESULTS mcc_opt --logx"
