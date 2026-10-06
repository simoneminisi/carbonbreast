#!/usr/bin/env bash
# Deployment matrix: {torch fp32, torch fp16, TensorRT fp16, TensorRT int8} on the GPU and
# {OpenVINO fp32, OpenVINO+NNCF int8} on the CPU, for the four frontier checkpoints, model-only
# (pre-decoded inputs, hardware power counters), batch 1 and 16, accuracy on the full test set.
# Cells already present in $RESULTS are skipped, so re-running is safe.
#
#   bash run_deploy.sh                          # everything (~1-2 h, mostly INT8 calibration + convnext CPU)
#   MODES=torch_fp32,torch_fp16 bash run_deploy.sh
#   BACKBONES="resnet18 efficientnet_b0" bash run_deploy.sh
#   LIMIT=64 LOOP_SECS=3 RESULTS=preflight.csv bash run_deploy.sh   # 10-min pre-flight of every mode
set -euo pipefail

DATA_DIR=${DATA_DIR:-../training_pytorch/}     # MUST end with /
RESULTS=${RESULTS:-results_deploy.csv}
MODES=${MODES:-torch_fp32,torch_fp16,trt_fp16,trt_int8,ov_fp32,ov_int8}
BATCHES=${BATCHES:-1,16}
LOOP_SECS=${LOOP_SECS:-20}
CALIB=${CALIB:-300}
NPROC=$(nproc)
WORKERS=${WORKERS:-$(( NPROC < 8 ? NPROC : 8 ))}
CPU_THREADS=${CPU_THREADS:-0}                       # 0 = OpenVINO default (all cores)
LIMIT=${LIMIT:-0}                                   # >0 = pre-flight on the first N test/val images only

[[ "$DATA_DIR" == */ ]] || { echo "DATA_DIR must end with / (got: $DATA_DIR)"; exit 1; }
[ -f "${DATA_DIR}png_index.csv" ] || { echo "missing ${DATA_DIR}png_index.csv"; exit 1; }

mkdir -p env
nvidia-smi -q > env/nvidia-smi.txt 2>&1 || echo "nvidia-smi unavailable" > env/nvidia-smi.txt
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > env/gpu.csv 2>/dev/null || true
lscpu > env/lscpu.txt 2>&1 || true
free -g > env/free.txt 2>&1 || true
python -c "import torch,tensorrt,openvino,nncf;print('torch',torch.__version__,'trt',tensorrt.__version__,'ov',openvino.__version__,'nncf',nncf.__version__)" | tee env/versions.txt
echo "results: $RESULTS | modes: $MODES | batches: $BATCHES | loop: ${LOOP_SECS}s | workers: $WORKERS | limit: $LIMIT"

# backbone            checkpoint                                  parent training row (tag in parent_results.csv)
CONFIGS=(
  "resnet18               checkpoints/best_resnet18_aug06.pt         resnet18_adamw_full_q-fp16_img768_bs16_ep15_r1_s42_trunk-1"
  "efficientnet_b0        checkpoints/best_efficientnet_b0.pt        efficientnet_b0_adamw_full_q-fp16_img768_bs24_ep15_r1_s42_trunk-0"
  "mobilenetv3_small_100  checkpoints/best_mobilenetv3_small_100.pt  mobilenetv3_small_100_adamw_full_q-fp16_img512_bs48_ep15_r1_s42_trunk-0"
  "convnext_small         checkpoints/best_convnext_small.pt         convnext_small_adamw_full_q-fp16_img768_bs8_ep15_r1_s42_trunk-1"
)
BACKBONES=${BACKBONES:-"resnet18 efficientnet_b0 mobilenetv3_small_100 convnext_small"}

for row in "${CONFIGS[@]}"; do
  read -r BB CKPT PTAG <<<"$row"
  [[ " $BACKBONES " == *" $BB "* ]] || continue
  [ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
  echo
  echo "===== $BB  ($PTAG) ====="
  python bench_infer.py --backbone "$BB" --ckpt "$CKPT" --parent-tag "$PTAG" --parent-results parent_results.csv \
    --folder "$DATA_DIR" --modes "$MODES" --batches "$BATCHES" --loop-secs "$LOOP_SECS" --calib-size "$CALIB" \
    --workers "$WORKERS" --cpu-threads "$CPU_THREADS" --results-csv "$RESULTS" --limit "$LIMIT"
done
echo
echo "DONE -> $RESULTS  (env/ holds nvidia-smi, lscpu, versions)"
