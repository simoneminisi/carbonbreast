#!/usr/bin/env bash
# Bare-TensorFlow replication of the pareto_mcc_opt frontier (7 Stage-2 runs + the 2 Stage-1
# trunks they depend on) -- the same 7 configs as run_pareto_keras.sh, without Keras/tf.data.
# Data (png_index.csv, patch_index.csv, processed_dataset_1024/, patches/) is read from DATA_DIR;
# the timm-converted ImageNet weights must be extracted into ./pretrained/ (see README).
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/../common" && pwd)${PYTHONPATH:+:$PYTHONPATH}"   # shared loader/metrics/energy modules

RESULTS=${RESULTS:-results_tf.csv}
DATA_DIR=${DATA_DIR:-../training_pytorch/}   # MUST end with /
PRETRAINED_DIR=${PRETRAINED_DIR:-pretrained}
SEED=${SEED:-42}
ROUNDS=${ROUNDS:-1}
NPROC=$(nproc)
WORKERS=${WORKERS:-$(( NPROC < 8 ? NPROC : 8 ))}
PREFETCH=${PREFETCH:-4}
XLA=${XLA:-0}                                       # XLA=1 -> jit_compile the train/predict steps
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}
XLA_ARGS=()
[ "$XLA" = 1 ] && XLA_ARGS=(--xla)
echo "framework: bare tensorflow  |  results: $RESULTS  |  data: $DATA_DIR  |  workers: $WORKERS  |  xla: $XLA"

[[ "$DATA_DIR" == */ ]] || { echo "DATA_DIR must end with / (got: $DATA_DIR)"; exit 1; }
[ -f "${DATA_DIR}png_index.csv" ] || { echo "missing ${DATA_DIR}png_index.csv"; exit 1; }
for BB in resnet18 mobilenetv3_small_100 convnext_small; do
  [ -f "${PRETRAINED_DIR}/${BB}.npz" ] || { echo "missing ${PRETRAINED_DIR}/${BB}.npz -- see ../pretrained/README.md"; exit 1; }
done

# ---------- Stage 1: patch trunks (skip if the trunk file already exists) ----------
for BB in resnet18 convnext_small; do
  TRUNK="trunk_tf_${BB}.npz"
  if [ -f "$TRUNK" ]; then
    echo "SKIP stage1 (exists): $TRUNK"
    continue
  fi
  echo "===== STAGE-1 TRUNK: $BB ====="
  python train_patch_tf.py --backbone "$BB" --folder "$DATA_DIR" --pretrained-dir "$PRETRAINED_DIR" \
    --img-size 224 --batch 64 --epochs 30 --warmup-epochs 3 --lr 3e-4 \
    --optimizer adamw --weight-decay 0.05 --seed "$SEED" --workers "$WORKERS" --prefetch "$PREFETCH" \
    --cache-ram --trunk-out "$TRUNK" "${XLA_ARGS[@]+"${XLA_ARGS[@]}"}"
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
    TRUNK_ARGS=(--pretrained-trunk "trunk_tf_${BB}.npz")
  fi
  echo "===== RUN: $TAG ====="
  python s_main_tf.py --backbone "$BB" "${TRUNK_ARGS[@]+"${TRUNK_ARGS[@]}"}" \
    --train-mode full --optimizer adamw --lr 1e-4 --weight-decay 0.05 --dropout 0.2 \
    --quantize fp16 --img-size "$IMG" --batch "$BS" --epochs "$EP" --warmup-epochs "$WU" \
    --rounds "$ROUNDS" --seed "$SEED" --workers "$WORKERS" --prefetch "$PREFETCH" \
    --folder "$DATA_DIR" --pretrained-dir "$PRETRAINED_DIR" --cache-ram --no-plot \
    --results-csv "$RESULTS" "${XLA_ARGS[@]+"${XLA_ARGS[@]}"}"
done

echo
echo "DONE. Plot with:  python ../common/plot_pareto.py $RESULTS mcc_opt --unit wh --logx --rowid"
