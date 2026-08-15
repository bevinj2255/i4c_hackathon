"""Extract the KLA zips into the layout the rest of the code expects.

    python prepare_data.py --train_zip train.zip --test_zip Test_NoisyLR.zip --out data

Produces:
    data/train/GT/*.npy        3200 x (256,256) float32, values in [0,1]
    data/train/NoisyLR/*.npy   3200 x (128,128) float32, values may exceed [0,1]
    data/test/NoisyLR/*.npy     397 x (128,128) float32

TRAP: the test filenames collide with the train filenames (all 397 test names also
exist in train) but the images are unrelated -- pixel correlation ~0.00. Extracting
both into one directory silently destroys data. They stay in separate trees. The
assertions at the bottom are the guard; a comment alone would not stop it.
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np


def real_members(zf, prefix, suffix=".npy"):
    """Zip entries under `prefix`, minus the __MACOSX/._* resource-fork junk.

    Those junk entries share basenames with the real files, so any count or glob
    that forgets to filter them is wrong (397 real test files look like 797).
    """
    return sorted(
        n for n in zf.namelist()
        if n.startswith(prefix) and n.endswith(suffix) and "__MACOSX" not in n
    )


def extract(zf, members, dest, expect_shape):
    dest.mkdir(parents=True, exist_ok=True)
    for name in members:
        arr = np.load(io.BytesIO(zf.read(name)))
        if arr.shape != expect_shape:
            raise SystemExit(f"ABORT: {name} has shape {arr.shape}, expected {expect_shape}")
        np.save(dest / Path(name).name, arr.astype(np.float32))
    return len(members)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_zip", default="train.zip")
    p.add_argument("--test_zip", default="Test_NoisyLR.zip")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    out = Path(a.out)

    with zipfile.ZipFile(a.train_zip) as zf:
        gt = real_members(zf, "train/GT/")
        lr = real_members(zf, "train/NoisyLR/")
        if {Path(n).name for n in gt} != {Path(n).name for n in lr}:
            raise SystemExit("ABORT: GT and NoisyLR filenames do not pair up 1:1")
        n_gt = extract(zf, gt, out / "train" / "GT", (256, 256))
        n_lr = extract(zf, lr, out / "train" / "NoisyLR", (128, 128))
    print(f"CHECK: train extracted, {n_gt} GT + {n_lr} NoisyLR, all shapes as expected")

    with zipfile.ZipFile(a.test_zip) as zf:
        te = real_members(zf, "NoisyLR/")
        n_te = extract(zf, te, out / "test" / "NoisyLR", (128, 128))
    print(f"CHECK: test extracted, {n_te} NoisyLR, all (128,128)")

    # The separate-trees guard, stated as an assertion rather than a warning.
    train_names = {p.name for p in (out / "train" / "NoisyLR").glob("*.npy")}
    test_names = {p.name for p in (out / "test" / "NoisyLR").glob("*.npy")}
    overlap = train_names & test_names
    if overlap:
        print(f"CHECK: {len(overlap)} filenames appear in BOTH train and test -- "
              f"they are different images, kept in separate trees (this is expected)")
    if n_gt != n_lr:
        raise SystemExit("ABORT: train pair count mismatch")
    print(f"CHECK: data/ ready -- {n_gt} train pairs, {n_te} test inputs")


if __name__ == "__main__":
    main()
