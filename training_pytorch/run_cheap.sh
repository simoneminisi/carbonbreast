#!/usr/bin/env bash
# Pareto step: populate the CHEAP (low training-carbon) end of the frontier, which is currently
# blank — every logged point so far is resnet18-or-bigger at 768px. Two levers:
#   * lighter backbones : mobilenetv3_small/large (1.5M/4.2M params), efficientnet_b0 (4.0M)
#   * lower resolution   : resnet18 @ 512 / 384 (carbon scales ~img_size^2)
# All full fine-tune, no trunk (ImageNet init — trunk doesn't help resnet18 and adds Stage-1 carbon),
# AdamW, fp16-quantize probe, so rows drop straight into results.csv next to the existing points.
# Each tag is checked and SKIPPED if already present, so re-run safely / add rows to grow the frontier.
#
#   bash run_cheap.sh
set -euo pipefail

EPOCHS=${EPOCHS:-15}; ROUNDS=${ROUNDS:-1}; SEED=${SEED:-42}
RESULTS=${RESULTS:-results.csv}
WORKERS=${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}

# config rows:  backbone  img_size  batch  lr
CONFIGS=(
  "mobilenetv3_small_100  768  32  1e-4"   # tiniest model @ full res
  "mobilenetv3_large_100  768  24  1e-4"
  "efficientnet_b0        768  24  1e-4"
  "resnet18               512  32  1e-4"   # resolution lever on the known knee
  "resnet18               384  48  1e-4"   # cheaper still
  "mobilenetv3_small_100  512  48  1e-4"   # extreme cheap corner: tiny model + low res
)

for row in "${CONFIGS[@]}"; do
  read -r BB IMG BS LR <<<"$row"
  tag="${BB}_adamw_full_q-fp16_img${IMG}_bs${BS}_ep${EPOCHS}_r${ROUNDS}_s${SEED}_trunk-0"
  if [ -f "$RESULTS" ] && grep -q ",${tag}," "$RESULTS"; then
    echo "SKIP (already in $RESULTS): $tag"; continue
  fi
  echo "===== RUN: $tag ====="
  python s_main.py --backbone "$BB" \
    --train-mode full --optimizer adamw --lr "$LR" --quantize fp16 \
    --img-size "$IMG" --batch "$BS" --workers "$WORKERS" \
    --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" \
    --cache-ram --no-plot --results-csv "$RESULTS"
done
echo "DONE — regenerate the training Pareto:  python ../common/plot_pareto.py $RESULTS mcc_opt --logx"
