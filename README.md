# CarbonBreast

Code for **CarbonBreast**, a study of accuracy versus energy for a mammography
classifier: how much accuracy each watt-hour buys when the same task is trained with different networks,
image sizes, optimisers, pre-training choices and software stacks, and what the trained models cost to serve.

*Paper: reference to be added on publication.*

The task is binary: does a mammogram look suspicious (BI-RADS 3–5) or not (BI-RADS 1–2), on the public
VinDr-Mammo dataset. The recipe is the patch-classifier transfer method of Shen et al. (2019) and Petrini and
Kim (2025): an optional stage that pre-trains the backbone on 224-px crops around annotated findings, then
whole-image fine-tuning. Every stage runs inside a CodeCarbon energy meter.

## What is in the repository

| folder | content |
|---|---|
| `preprocessing/` | `preprocess.py` (DICOM → 1,024-px 8-bit PNG: Otsu crop, CLAHE, pectoral removal, right-facing flip, padding) and `build_index.py` (binary label and split index) |
| `training_pytorch/` | the reference pipeline: `make_patches.py` and `train_patch.py` (stage-1), `s_main.py` and `pipeline.py` (stage-2, evaluation, energy regions), and the `run_*.sh` scripts that produced the 23-run pool |
| `training_keras/` | the same seven frontier configurations in Keras 3 on the TensorFlow or JAX backend |
| `training_tf/`, `training_jax/` | Keras-free ports in plain TensorFlow (`XLA=0/1`) and plain JAX (flax + optax); their run scripts put `../common` on `PYTHONPATH` |
| `common/` | framework-free code shared by the two ports (numpy/OpenCV loader, metrics, energy regions, results schema, LR schedule, npz I/O, the timm-to-npz converter, the Pareto plot) and its tests |
| `pretrained/` | manifest of the timm weights converted for the two ports (files distributed as a release asset, see its README) |
| `deployment/` | `bench_infer.py` and `run_deploy.sh`: model-only serving benchmark on GPU (PyTorch fp32/fp16, TensorRT fp16/int8) and CPU (OpenVINO fp32, NNCF int8) |

Not included: the dataset (credentialed), trained checkpoints and trunks, the converted pretrained weights
(release asset), the results files and figures (reported in the paper), and the separate from-scratch
replication of Petrini and Kim that preceded this work (it vendors the authors' code).

## Data

1. Obtain VinDr-Mammo from PhysioNet (credentialed access): https://physionet.org/content/vindr-mammo/
2. Place it so that `$CARBONBREAST_DATA/VINDR-MAMMO/` holds `images/`, `finding_annotations.csv` and
   `breast-level_annotations.csv` (default `CARBONBREAST_DATA=<repo>/data`).
3. Preprocess once (CPU-bound, ~20k images):

```bash
python preprocessing/preprocess.py --dataset-dir $CARBONBREAST_DATA/VINDR-MAMMO \
       --out-root $CARBONBREAST_DATA/processed_dataset_1024 --target-size 1024 --workers 8
python preprocessing/build_index.py            # -> png_index.csv (label, split, exam id per image)
```

The split is deterministic (official VinDr test split; validation = 15% of training exams, GroupShuffleSplit with seed 42).

## Environments

One virtual environment per framework; their CUDA wheels conflict and must never share an environment.

| environment | install |
|---|---|
| PyTorch (training, analysis) | `pip install torch torchvision` for your CUDA, then `pip install -r training_pytorch/requirements.txt` |
| Keras 3 on TensorFlow | `pip install -r training_keras/requirements_keras.txt` |
| Keras 3 on JAX | `pip install -r training_keras/requirements_keras_jax.txt` |
| plain TensorFlow | `pip install -r training_tf/requirements_tf.txt` |
| plain JAX | `pip install -r training_jax/requirements_jax.txt` |
| serving benchmark | `pip install -r deployment/requirements_deploy.txt` (pins `tensorrt-cu12`; the bare `tensorrt` package resolves to CUDA 13 builds) |

The reported runs were made on rented cloud instances with one NVIDIA RTX 4080 SUPER; the serving benchmark
additionally used the host's AMD EPYC 7K62 CPU (23-core quota). Energy figures depend on the hardware.

## Reproducing the training runs

```bash
cd training_pytorch
python make_patches.py                                   # stage-1 patches -> patches/, patch_index.csv
python train_patch.py --backbone resnet18                # stage-1 trunk  -> trunk_resnet18.pt (+ .energy.json)
bash run_pareto.sh                                       # the frontier batch (skips tags already in results.csv)
bash run_cheap.sh; bash run_large.sh; bash run_next.sh; bash run_longsched.sh   # the rest of the 23-run pool
python s_main.py --backbone resnet18 --img-size 768 --batch 16 --epochs 15 \
       --pretrained-trunk trunk_resnet18.pt --quantize fp16 --no-plot --cache-ram      # one run by hand
```

Each run appends one 49-column row to `results.csv` under a tag that encodes its configuration, writes
`predictions/pred_<tag>.csv`, and saves the best-validation checkpoint. The same tags are used by every framework.

Framework replications (seven configurations each, stage-1 retrained in the framework):

```bash
cd training_keras && DATA_DIR=../training_pytorch/ bash run_pareto_keras.sh                 # Keras 3, TensorFlow
cd training_keras && DATA_DIR=../training_pytorch/ BACKEND=jax bash run_pareto_keras.sh     # Keras 3, JAX
cd training_tf    && DATA_DIR=../training_pytorch/ bash run_pareto_tf.sh                    # plain TensorFlow
cd training_tf    && DATA_DIR=../training_pytorch/ XLA=1 RESULTS=results_tf_xla.csv bash run_pareto_tf.sh
cd training_jax   && DATA_DIR=../training_pytorch/ bash run_pareto_jax.sh                   # plain JAX
```

The two plain ports need the converted weights in `<port>/pretrained/` (see `pretrained/README.md`). To run their parity
tests or any module by hand, add the shared code to the path: `PYTHONPATH=../common python parity_test_tf.py --all`.

## Reproducing the serving benchmark

```bash
cd deployment
# put the four checkpoints in checkpoints/ (names and parent run tags are listed in run_deploy.sh)
LIMIT=64 LOOP_SECS=3 RESULTS=preflight.csv DATA_DIR=../training_pytorch/ bash run_deploy.sh   # 10-minute pre-flight
CPU_THREADS=23 DATA_DIR=../training_pytorch/ bash run_deploy.sh                                # full matrix, 1-2 h
```

## Headline results

- More energy buys little accuracy: resnet18 at 512 px reaches AUROC 0.772 for 35 Wh; the best per-image AUROC
  (0.796) costs 112 Wh at 768 px; a 13× larger budget does not move it.
- The software stack changes the training bill by up to 3× for the same model and data; the XLA compiler made
  TensorFlow faster but not cheaper.
- MCC at a validation-chosen threshold wobbles by ±0.04 between repeated runs; AUROC by ±0.01.
- For serving, TensorRT fp16 on the GPU is 3.7–5.6× faster than fp32 at identical accuracy; int8 helps some
  architectures and breaks others; the CPU costs 22–40× the energy per image of the best GPU setting.

Details, tables and the reasoning behind each number are in the paper.

## Notes on provenance

- The code in `training_pytorch/` names checkpoints per run tag. The reported PyTorch batches were produced by an
  earlier version that named them `best_<backbone>.pt`, so later runs of the same backbone overwrote earlier
  checkpoints; metrics and energies are unaffected, only which checkpoint file survived on disk.
- The only edits made to the code after the runs are path handling: data locations read `CARBONBREAST_DATA`, the
  patch script imports the preprocessing module from `preprocessing/`, and the port and benchmark scripts default to
  `../training_pytorch/` as data folder.
- `deployment/parent_results.csv` holds the training rows of the four served checkpoints; the benchmark reads it to
  cross-check that each checkpoint reproduces its training AUROC.

## Citation

See `CITATION.cff`. The dataset is VinDr-Mammo (Nguyen et al., *Scientific Data* 10, 277, 2023); the training recipe
follows Shen et al. (*Scientific Reports* 9, 12495, 2019) and Petrini and Kim (arXiv:2503.19945, 2025); energy is
measured with CodeCarbon.

## License

MIT, see `LICENSE`.
