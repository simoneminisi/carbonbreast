#!/usr/bin/env bash
# Longer-schedule test: are the larger models under-trained at 15 epochs, or genuinely at the
# data-limited ceiling? Re-run the ambiguous ones with a stretched schedule and compare TEST AUROC
# to their 15-epoch rows (already in results.csv). Everything else matches the original runs, so
# epochs (and, for ViT, LR) is the only thing that changes:
#   convnext_small  40ep / warmup 5 / patch-trunk / bs8   (vs 15ep 0.808 breast — the trunk-1 point)
#   convnext_base   40ep / warmup 5 / no-trunk   / bs16   (vs 15ep 0.784)
#   vit_base        50ep / warmup 8 / lr 5e-5 / grad-ckpt / bs16  (vs 15ep 0.739; ViT wants long+low-LR)
#
# Per-tag checkpoints (best_<tag>.pt) mean these NEVER clobber your existing models, and the longer
# epoch count makes each a NEW tag, so they log as fresh points (old 15ep rows untouched). The
# skip-if-done guard lets you re-run safely.
#
#   bash run_longsched.sh
# NOTE: vit_base @768 for 50 epochs with grad-checkpointing is the slow one (several hours on a
# single GPU) — expected. Sized for a 32 GB card; drop a batch or add --grad-checkpoint if OOM.
set -euo pipefail

IMG=${IMG:-768}; ROUNDS=${ROUNDS:-1}; SEED=${SEED:-42}
RESULTS=${RESULTS:-results.csv}
WORKERS=${WORKERS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}

# config rows:  backbone  batch  epochs  warmup  use_trunk(0/1)  extra_flags
CONFIGS=(
  "convnext_small        8   40  5  1  "
  "convnext_base         16  40  5  0  "
  "vit_base_patch16_224  16  50  8  0  --lr 5e-5 --grad-checkpoint"
)

for row in "${CONFIGS[@]}"; do
  read -r BB BS EPOCHS WARMUP USE_TRUNK EXTRA <<<"$row"
  tag="${BB}_adamw_full_q-fp16_img${IMG}_bs${BS}_ep${EPOCHS}_r${ROUNDS}_s${SEED}_trunk-${USE_TRUNK}"
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
    --train-mode full --optimizer adamw --quantize fp16 $EXTRA \
    --img-size "$IMG" --batch "$BS" --workers "$WORKERS" \
    --epochs "$EPOCHS" --warmup-epochs "$WARMUP" --rounds "$ROUNDS" --seed "$SEED" \
    --cache-ram --no-plot --results-csv "$RESULTS"
done
echo "DONE — compare vs the 15ep rows:  python ../common/plot_pareto.py $RESULTS mcc_opt --unit wh --logx"
