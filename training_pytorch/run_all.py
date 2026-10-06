"""Run the full 2-stage pipeline end-to-end.

    Stage 0: build_index.py      -> png_index.csv        (full-image labels/splits)
    Stage 1: make_patches.py     -> patches/, patch_index.csv
             train_patch.py      -> patch_trunk.pt       (lesion-vs-background conv trunk)
    Stage 2: s_main.py --pretrained-trunk patch_trunk.pt -> best_png.pt + test metrics

Use --no-patch-pretrain to skip Stage 1 and train from plain ImageNet init (the A/B baseline
for measuring what the patch-pretraining transfer is worth).

    python run_all.py                      # full pipeline (patch-pretrained)
    python run_all.py --no-patch-pretrain  # ImageNet baseline
    python run_all.py --extra "--epochs 40 --img-size 768"
"""
import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-patch-pretrain", dest="patch_pretrain", action="store_false")
    ap.add_argument("--skip-patches", action="store_true", help="reuse existing patch_index.csv")
    ap.add_argument("--backbone", default="resnet18",
                    help="threaded to BOTH stages (torchvision or timm); must match end to end")
    ap.add_argument("--extra", default="", help="extra args forwarded to s_main.py")
    args = ap.parse_args()
    py = sys.executable
    trunk = f"trunk_{args.backbone}.pt"

    if not os.path.exists("png_index.csv"):
        run([py, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing", "build_index.py")])

    trunk_args = []
    if args.patch_pretrain:
        if not args.skip_patches:
            run([py, "make_patches.py"])
        run([py, "train_patch.py", "--backbone", args.backbone, "--trunk-out", trunk])
        trunk_args = ["--pretrained-trunk", trunk]

    run([py, "s_main.py", "--backbone", args.backbone, *trunk_args, *args.extra.split()])


if __name__ == "__main__":
    main()
