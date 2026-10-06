"""Command-line interfaces reproducing s_main_keras.py / train_patch_keras.py (same flags, same
defaults) plus the few options the bare ports add: --pretrained-dir, --no-pretrained,
--prefetch and (TF only) --xla."""
import argparse

BACKBONES = ("resnet18", "mobilenetv3_small_100", "convnext_small")


def stage2_parser(description, default_results, trunk_hint, extra=None):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="", help="prefix prepended to png_index.csv and png paths")
    p.add_argument("--png-index", default=None, help="path to png_index.csv (default: <folder>png_index.csv)")
    p.add_argument("--backbone", default="resnet18", choices=BACKBONES)
    p.add_argument("--pretrained-trunk", default=None, help=f"{trunk_hint} from the Stage-1 script")
    p.add_argument("--img-size", type=int, default=1024)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--no-amp", dest="amp", action="store_false", help="train in fp32")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4, help="peak LR (cosine decays from here)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--momentum", type=float, default=0.9, help="for sgd/rmsprop")
    p.add_argument("--train-mode", default="full", choices=["full", "head"],
                   help="full = fine-tune everything; head = freeze backbone, train only the classifier")
    p.add_argument("--quantize", default="none", choices=["none", "fp16"],
                   help="fp16 = also evaluate a half-precision copy and report size/AUROC/latency")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.2, help="dropout before the classifier (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=1, help="train N seeded rounds, keep best-on-val")
    p.add_argument("--workers", type=int, default=8, help="loader worker processes (0 = inline)")
    p.add_argument("--prefetch", type=int, default=4, help="extra ring-buffer batches per loader")
    p.add_argument("--cache-ram", action="store_true", help="preload train+val decoded into RAM")
    p.add_argument("--cache-test", action="store_true", help="also cache the test set")
    p.add_argument("--cache-workers", type=int, default=None, help="decode threads (default: --workers)")
    p.add_argument("--force-cache", action="store_true", help="bypass the RAM-fits guard")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon tracking")
    p.add_argument("--ckpt", default=None, help="checkpoint path (default: best_<tag>.npz)")
    p.add_argument("--pred-dir", default="predictions", help="dir for per-image prediction CSVs")
    p.add_argument("--results-csv", default=default_results, help="append one summary row per run here")
    p.add_argument("--no-plot", action="store_true", help="save confusion-matrix PNGs (headless)")
    p.add_argument("--limit", type=int, default=None,
                   help="DEBUG: stratified subset of N rows per split (smoke tests only)")
    p.add_argument("--pretrained-dir", default="pretrained",
                   help="dir with the timm-converted <backbone>.npz ImageNet weights")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    if extra:
        extra(p)
    return p


def stage1_parser(description, default_trunk_hint, extra=None):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="", help="prefix prepended to patch_index.csv and patch paths")
    p.add_argument("--patch-index", default=None, help="default: <folder>patch_index.csv")
    p.add_argument("--backbone", default="resnet18", choices=BACKBONES)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4, help="peak LR (cosine decays from here)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd", "rmsprop"])
    p.add_argument("--momentum", type=float, default=0.9, help="for sgd/rmsprop")
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8, help="loader worker processes (0 = inline)")
    p.add_argument("--prefetch", type=int, default=4, help="extra ring-buffer batches per loader")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-energy", dest="energy", action="store_false", help="disable CodeCarbon tracking")
    p.add_argument("--cache-ram", action="store_true", help="preload patches decoded into RAM")
    p.add_argument("--force-cache", action="store_true", help="bypass the RAM-fits guard")
    p.add_argument("--trunk-out", default=None, help=f"default: {default_trunk_hint}")
    p.add_argument("--limit", type=int, default=None,
                   help="DEBUG: stratified subset of N rows per split (smoke tests only)")
    p.add_argument("--pretrained-dir", default="pretrained",
                   help="dir with the timm-converted <backbone>.npz ImageNet weights")
    p.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    if extra:
        extra(p)
    return p


def add_xla_flag(p):
    p.add_argument("--xla", action="store_true", help="compile the train/predict steps with XLA (jit_compile)")
