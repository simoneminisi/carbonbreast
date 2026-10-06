#!/usr/bin/env bash
# First-pass subset to sketch an accuracy-vs-carbon Pareto curve.
# Spreads points along backbone size x train-mode x (trunk / no-trunk). Every run appends to
# results.csv with per-breast AUROC + energy/CO2 for Stage-1, Stage-2 and inference.
#
#   bash run_pareto.sh            # 768px, 15 epochs, rounds 1
set -euo pipefail

IMG=${IMG:-768}
EPOCHS=${EPOCHS:-15}
ROUNDS=${ROUNDS:-1}
# Cap workers: with --cache-ram a few workers saturate the GPU, and huge --workers (e.g. a
# 192-vCPU box) floods DataLoader IPC -> "bus error / insufficient shared memory". 8 is plenty.
WORKERS=${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}

# config rows:  backbone  use_trunk(0/1)  train-mode  batch
CONFIGS=(
  "resnet18        0 head 16"
  "resnet18        0 full 16"
  "resnet18        1 head 16"
  "resnet18        1 full 16"
  "convnext_tiny   1 head 16"
  "convnext_tiny   1 full 16"
  "convnext_small  1 head 8"
  "convnext_small  1 full 8"
)

lr_for() { [ "$1" = head ] && echo 0.001 || echo 0.0001; }  # linear-probe wants a higher LR

# Stage 1: train the patch trunk once for each backbone that any config uses with a trunk
for BB in $(printf '%s\n' "${CONFIGS[@]}" | awk '$2==1{print $1}' | sort -u); do
  [ -f "trunk_${BB}.pt" ] || { echo "===== Stage 1: patch trunk $BB ====="; \
    python train_patch.py --backbone "$BB" --trunk-out "trunk_${BB}.pt" --cache-ram; }
done

# Stage 2: one run per config
for row in "${CONFIGS[@]}"; do
  read -r BB USE_TRUNK MODE BS <<<"$row"
  LR=$(lr_for "$MODE")
  TRUNK_ARG=(); [ "$USE_TRUNK" = 1 ] && TRUNK_ARG=(--pretrained-trunk "trunk_${BB}.pt")
  echo "===== $BB | trunk=$USE_TRUNK | $MODE | bs=$BS lr=$LR ====="
  python s_main.py --backbone "$BB" "${TRUNK_ARG[@]}" \
    --train-mode "$MODE" --optimizer adamw --lr "$LR" --quantize fp16 \
    --img-size "$IMG" --batch "$BS" --workers "$WORKERS" \
    --epochs "$EPOCHS" --rounds "$ROUNDS" --cache-ram --no-plot
done
echo "DONE — build the Pareto from results.csv (see ../common/plot_pareto.py)"
