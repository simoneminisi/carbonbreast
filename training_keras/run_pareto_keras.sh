#!/usr/bin/env bash
# Keras replication of the pareto_mcc_opt frontier (7 Stage-2 runs + the 2 Stage-1 trunks
# they depend on). Data (png_index.csv, patch_index.csv, processed_dataset_1024/, patches/)
# is read from DATA_DIR (default: the original experiment dir extracted next to this one).
#
# Backend: BACKEND=tensorflow (default) or BACKEND=jax. Each backend writes its own
# results CSV (results_keras.csv / results_keras_jax.csv) -- the run tags are identical
# across backends, the CSV file is what distinguishes the framework.
set -euo pipefail

BACKEND=${BACKEND:-tensorflow}
export KERAS_BACKEND="$BACKEND"
if [ "$BACKEND" = jax ]; then
  RESULTS=${RESULTS:-results_keras_jax.csv}
else
  RESULTS=${RESULTS:-results_keras.csv}
fi
DATA_DIR=${DATA_DIR:-../training_pytorch/}   # MUST end with /
SEED=${SEED:-42}
ROUNDS=${ROUNDS:-1}
NPROC=$(nproc)
WORKERS=${WORKERS:-$(( NPROC < 8 ? NPROC : 8 ))}
echo "backend: $BACKEND  |  results: $RESULTS  |  data: $DATA_DIR"

[[ "$DATA_DIR" == */ ]] || { echo "DATA_DIR must end with / (got: $DATA_DIR)"; exit 1; }
[ -f "${DATA_DIR}png_index.csv" ] || { echo "missing ${DATA_DIR}png_index.csv"; exit 1; }

# ---------- Stage 1: patch trunks (skip if the trunk file already exists) ----------
for BB in resnet18 convnext_small; do
  TRUNK="trunk_keras_${BB}.weights.h5"
  if [ -f "$TRUNK" ]; then
    echo "SKIP stage1 (exists): $TRUNK"
    continue
  fi
  echo "===== STAGE-1 TRUNK: $BB ====="
  python train_patch_keras.py --backbone "$BB" --folder "$DATA_DIR" \
    --img-size 224 --batch 64 --epochs 30 --warmup-epochs 3 --lr 3e-4 \
    --optimizer adamw --weight-decay 0.05 --seed "$SEED" --workers "$WORKERS" \
    --cache-ram --trunk-out "$TRUNK"
done

# ---------- Stage 2: the 7 pareto_mcc_opt frontier configs ----------
#          backbone               img  bs  ep  warmup trunk
CONFIGS=(
  "resnet18                384  48  15  3  0"
  "mobilenetv3_small_100   512  48  15  3  0"
  "resnet18                512  32  15  3  0"
  "mobilenetv3_small_100   768  32  15  3  0"
  "resnet18                768  16  15  3  0"
  "resnet18                768  16  15  3  1"
  "convnext_small          768   8  40  5  1"
)

for row in "${CONFIGS[@]}"; do
  read -r BB IMG BS EP WU TR <<<"$row"
  TAG="${BB}_adamw_full_q-fp16_img${IMG}_bs${BS}_ep${EP}_r${ROUNDS}_s${SEED}_trunk-${TR}"
  if [ -f "$RESULTS" ] && grep -q ",${TAG}," "$RESULTS"; then
    echo "SKIP (already in $RESULTS): $TAG"
    continue
  fi
  TRUNK_ARGS=()
  if [ "$TR" = 1 ]; then
    TRUNK_ARGS=(--pretrained-trunk "trunk_keras_${BB}.weights.h5")
  fi
  echo "===== RUN: $TAG ====="
  python s_main_keras.py --backbone "$BB" "${TRUNK_ARGS[@]+"${TRUNK_ARGS[@]}"}" \
    --train-mode full --optimizer adamw --lr 1e-4 --weight-decay 0.05 --dropout 0.2 \
    --quantize fp16 --img-size "$IMG" --batch "$BS" --epochs "$EP" --warmup-epochs "$WU" \
    --rounds "$ROUNDS" --seed "$SEED" --workers "$WORKERS" \
    --folder "$DATA_DIR" --cache-ram --no-plot --results-csv "$RESULTS"
done

echo
echo "DONE. Plot with:  python ../common/plot_pareto.py $RESULTS mcc_opt --unit wh --logx --rowid"
