"""V1 -- loader determinism / shape / balance checks (numpy only, runs in any venv).

    python test_pipeline_np.py --folder ../training_pytorch/ [--n 96 --img-size 96]
"""
import argparse
import time

import numpy as np
import pandas as pd

from pipeline_np import (
    EvalLoader, TrainLoader, WorkerPool, balanced_probs, decode_all, limit_df, split_trainval,
    zscore_3ch_hwc,
)


def collect(loader, steps=None):
    xs, ys = [], []
    for k, (x, y) in enumerate(loader):
        xs.append(x.copy())
        ys.append(np.asarray(y).copy())
        if steps and k + 1 >= steps:
            break
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="../training_pytorch/")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    df = pd.read_csv(a.folder + "png_index.csv")
    train_df, val_df = split_trainval(df)
    train_df, val_df = limit_df(train_df, a.n), limit_df(val_df, a.n)
    cache = decode_all(train_df, a.img_size, 4, a.folder, "test cache")
    vcache = decode_all(val_df, a.img_size, 4, a.folder, "test val cache")

    # (i) determinism: 2 workers vs inline, same seed -> bitwise identical batches
    pool2, pool0 = WorkerPool(2), WorkerPool(0)
    l2 = TrainLoader(train_df, a.img_size, a.batch, 42, a.folder, cache, pool2)
    l0 = TrainLoader(train_df, a.img_size, a.batch, 42, a.folder, cache, pool0)
    ev = EvalLoader(val_df, a.img_size, a.batch, a.folder, vcache, pool2)  # all loaders BEFORE start()
    pool2.start(); pool0.start()
    try:
        EvalLoader(val_df, a.img_size, a.batch, a.folder, vcache, pool2)
        raise AssertionError("loader after start() must be rejected")
    except RuntimeError:
        print("(0) post-start loader construction correctly rejected")
    x2, y2 = collect(l2)
    x0, y0 = collect(l0)
    assert len(x2) == len(x0) == len(train_df) // a.batch, "epoch length must be n//batch"
    for i, (p, q) in enumerate(zip(x2, x0)):
        assert np.array_equal(p, q), f"batch {i} differs between 2 workers and inline"
    for p, q in zip(y2, y0):
        assert np.array_equal(p, q)
    assert x2[0].shape == (a.batch, a.img_size, a.img_size, 3) and x2[0].dtype == np.float32
    print(f"(i) determinism OK: {len(x2)} batches bitwise identical (2 workers vs inline)")

    # second epoch differs from the first (stream advances) but is itself deterministic
    x2b, _ = collect(l2)
    x0b, _ = collect(l0)
    assert all(np.array_equal(p, q) for p, q in zip(x2b, x0b))
    assert not np.array_equal(x2[0], x2b[0]), "second epoch must draw different indices/augs"
    print("(i') epoch-2 determinism OK")

    # (ii) balanced index stream: class frequencies ~ equal over many draws
    labels = train_df["label"].to_numpy()
    p = balanced_probs(labels)
    rng = np.random.default_rng(0)
    draws = rng.choice(len(labels), size=20000, replace=True, p=p)
    frac1 = labels[draws].mean()
    assert 0.45 < frac1 < 0.55, f"balanced sampler gave class-1 fraction {frac1:.3f}"
    print(f"(ii) balanced sampling OK: class-1 fraction {frac1:.3f} (raw {labels.mean():.3f})")

    # (iii) eval loader == serial z-score, fixed shapes, correct n_valid
    rows, n_seen = [], 0
    for x, nv in ev:
        assert x.shape == (a.batch, a.img_size, a.img_size, 3)
        rows.append(x[:nv].copy()); n_seen += nv
    got = np.concatenate(rows)
    assert n_seen == len(val_df) and got.shape[0] == len(val_df)
    ref = np.stack([zscore_3ch_hwc(vcache[i], a.img_size) for i in range(len(val_df))])
    assert np.array_equal(got, ref), "eval loader output != serial z-score"
    print(f"(iii) eval loader OK: {len(val_df)} rows, {len(ev)} fixed-shape batches")

    # (iv) throughput (informational)
    t0 = time.time(); nb = 0
    for _ in l2:
        nb += 1
    dt = time.time() - t0
    print(f"(iv) throughput @ {a.img_size}px bs{a.batch} 2 workers: {nb * a.batch / dt:.0f} img/s")
    pool2.close(); pool0.close()
    print("ALL LOADER TESTS PASSED")


if __name__ == "__main__":
    main()
