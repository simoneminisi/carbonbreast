#!/usr/bin/env bash
# Pareto step: probe the HIGH-accuracy / high-carbon end with larger backbones — a "ceiling probe".
# The task ceiling looks data-limited (~0.81 breast AUROC, only ~1.3k positive train images) and
# convnext_tiny already OVERFIT (val 0.821 / test 0.787), so these are genuinely uncertain: they may
# beat convnext_small (0.808) or just cost more carbon for nothing. Three points answer it:
#   resnet50 (23M, trunk-compatible upgrade from resnet18) | convnext_base (88M) | vit_base (86M)
# All full fine-tune, no trunk (ViT pos-embeds are resolution-specific; resnet50 has no trunk built).
# Sized for a 32 GB GPU at 768px WITHOUT grad-checkpointing (keeps training carbon honest — no
# recompute). If you hit OOM on a smaller card, append --grad-checkpoint (timm models) or lower the
# batch in the row. Rows land in results.csv next to the cheap-end + existing points.
#
#   bash run_large.sh
set -euo pipefail

IMG=${IMG:-768}; EPOCHS=${EPOCHS:-15}; ROUNDS=${ROUNDS:-1}; SEED=${SEED:-42}
RESULTS=${RESULTS:-results.csv}
WORKERS=${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}

# config rows:  backbone  batch  lr  extra_flags
CONFIGS=(
  "resnet50               32  1e-4  "                    # cheapest 'bigger' point
  "convnext_base          16  1e-4  "                    # does more conv capacity beat convnext_small?
  "vit_base_patch16_224   16  1e-4  "                    # transformer capacity (dynamic@768, no trunk)
)

for row in "${CONFIGS[@]}"; do
  read -r BB BS LR EXTRA <<<"$row"
  tag="${BB}_adamw_full_q-fp16_img${IMG}_bs${BS}_ep${EPOCHS}_r${ROUNDS}_s${SEED}_trunk-0"
  if [ -f "$RESULTS" ] && grep -q ",${tag}," "$RESULTS"; then
    echo "SKIP (already in $RESULTS): $tag"; continue
  fi
  echo "===== RUN: $tag ====="
  python s_main.py --backbone "$BB" \
    --train-mode full --optimizer adamw --lr "$LR" --quantize fp16 $EXTRA \
    --img-size "$IMG" --batch "$BS" --workers "$WORKERS" \
    --epochs "$EPOCHS" --rounds "$ROUNDS" --seed "$SEED" \
    --cache-ram --no-plot --results-csv "$RESULTS"
done
echo "DONE — regenerate the training Pareto:  python ../common/plot_pareto.py $RESULTS mcc_opt --logx"
