"""Training data: half real pairs, half freshly synthesized ones.

The mix is the point. The 3200 provided pairs carry exactly one noise realisation
each, so training on them alone lets the network memorise those particular noise
fields. Because we recovered the forward model (see src/degrade.py) we can also
manufacture a brand new degraded input from any clean GT on demand, with noise
levels drawn from a range wider than the one we measured.

  - real pairs keep us honest about the actual test distribution
  - synthetic pairs give unlimited fresh noise and wider noise levels, which is what
    the out-of-distribution half of the hidden test set rewards

Geometric augmentation is the 8-element dihedral group (4 rotations x optional flip).

A note on why, because the obvious justification is wrong: the training images are
NOT orientation-free. Inspect them (results/figures/dataset_sample.png) and they turn
out to be generic grayscale natural photographs -- books, foliage, buildings, printed
text -- which very much have a canonical "up". The augmentation is still correct, but
for a different reason: the task is a LOCAL restoration operator, and both the
degradation (per-pixel noise, block averaging) and its inverse are equivariant under
the dihedral group. Rotating an image rotates its ideal restoration exactly. So the
extra orientations are valid training signal even though the content is not isotropic.

It also commutes with the 2x2 area downsample, so for synthetic samples we can rotate
the GT first and degrade afterwards and get the same distribution.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import synthesize

VAL_COUNT = 200
SPLIT_SEED = 0


def split_names(gt_dir, val_count=VAL_COUNT, seed=SPLIT_SEED):
    """Deterministic train/val split. Val is never trained on and never tuned into.

    Sorted first so the split does not depend on filesystem ordering, then shuffled
    with a fixed seed so the two sets are not correlated with the file numbering.
    """
    names = sorted(p.name for p in Path(gt_dir).glob("*.npy"))
    if not names:
        raise SystemExit(f"ABORT: no .npy files in {gt_dir} -- run prepare_data.py first")
    idx = np.random.default_rng(seed).permutation(len(names))
    val = {names[i] for i in idx[:val_count]}
    return [n for n in names if n not in val], [n for n in names if n in val]


def dihedral(arr, k):
    """One of the 8 symmetries of the square. k in 0..7."""
    if k & 4:
        arr = arr[:, ::-1]
    return np.rot90(arr, k & 3)


class RestorationDataset(Dataset):
    def __init__(self, names, gt_dir, lr_dir, patch=64, scale=2,
                 p_synth=0.5, augment=True, seed=0):
        self.names = list(names)
        self.gt_dir, self.lr_dir = Path(gt_dir), Path(lr_dir)
        self.patch, self.scale = patch, scale
        self.p_synth, self.augment = p_synth, augment
        # One generator per dataset instance. With persistent workers this advances
        # across epochs, so every epoch sees different noise while the run as a whole
        # stays reproducible from `seed`.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        gt = np.load(self.gt_dir / name)

        if self.augment:
            k = int(self.rng.integers(0, 8))
            gt = np.ascontiguousarray(dihedral(gt, k))

        if self.rng.random() < self.p_synth:
            lr, _, _ = synthesize(gt, self.rng, self.scale)
        else:
            lr = np.load(self.lr_dir / name)
            if self.augment:
                lr = np.ascontiguousarray(dihedral(lr, k))

        if self.patch:
            lr, gt = self._crop(lr, gt)
        return (torch.from_numpy(lr.copy()).unsqueeze(0),
                torch.from_numpy(gt.copy()).unsqueeze(0))

    def _crop(self, lr, gt):
        """Aligned random crop: LR at (y,x) size p, GT at (scale*y, scale*x) size scale*p."""
        p, s = self.patch, self.scale
        h, w = lr.shape
        if h <= p or w <= p:
            return lr, gt
        y = int(self.rng.integers(0, h - p + 1))
        x = int(self.rng.integers(0, w - p + 1))
        return (lr[y:y + p, x:x + p],
                gt[y * s:(y + p) * s, x * s:(x + p) * s])


class EvalDataset(Dataset):
    """Full images, no augmentation, real pairs only -- what validation must measure."""

    def __init__(self, names, gt_dir, lr_dir):
        self.names = list(names)
        self.gt_dir, self.lr_dir = Path(gt_dir), Path(lr_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        return (torch.from_numpy(np.load(self.lr_dir / n)).unsqueeze(0),
                torch.from_numpy(np.load(self.gt_dir / n)).unsqueeze(0))


def _demo():
    """Crops must stay aligned, or the model learns a shifted mapping."""
    import tempfile
    from .degrade import area_downsample

    with tempfile.TemporaryDirectory() as d:
        gt_dir, lr_dir = Path(d) / "GT", Path(d) / "LR"
        gt_dir.mkdir(); lr_dir.mkdir()
        yy, xx = np.mgrid[0:256, 0:256] / 255.0
        gt = ((np.sin(xx * 9) * np.cos(yy * 7) + 1) / 2).astype(np.float32)
        for i in range(4):
            np.save(gt_dir / f"{i:06d}.npy", gt)
            np.save(lr_dir / f"{i:06d}.npy", area_downsample(gt))

        # p_synth=0 forces the real-pair path, where LR is a clean area-downsample of
        # GT -- so an aligned crop must satisfy area_downsample(gt_crop) == lr_crop.
        ds = RestorationDataset([f"{i:06d}.npy" for i in range(4)], gt_dir, lr_dir,
                                patch=32, p_synth=0.0, augment=True, seed=3)
        for i in range(4):
            lr, g = ds[i]
            assert lr.shape == (1, 32, 32) and g.shape == (1, 64, 64), (lr.shape, g.shape)
            err = np.abs(area_downsample(g[0].numpy()) - lr[0].numpy()).max()
            assert err < 1e-5, f"crop misalignment: max err {err}"

        tr, va = split_names(gt_dir, val_count=1)
        assert len(tr) == 3 and len(va) == 1 and not (set(tr) & set(va))
        assert split_names(gt_dir, val_count=1)[1] == va, "split is not deterministic"

    print("CHECK: data.py self-check passed (crops aligned under augmentation, "
          "split deterministic and disjoint)")


if __name__ == "__main__":
    _demo()
