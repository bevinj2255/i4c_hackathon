"""Randomised synthetic structure, for optionally broadening the training content.

The KLA training set is natural photography (results/figures/dataset_sample.png), yet
the hidden test set is stated to contain out-of-distribution content and the task is
framed around semiconductor inspection. Measured with ood_check.py, the model trained on
natural images alone is weakest exactly where wafer imagery lives: dense periodic
structure and hard step edges.

This generates the *kind* of structure that domain is made of -- gratings, contact
arrays, checkerboards, step edges -- with randomised pitch, angle, duty cycle, contrast
and offset, so a model can be trained across it rather than on a fixed handful.

Deliberately NOT claimed to be real wafer imagery. It is generic periodic and
high-contrast structure, used to stop the network assuming everything looks like a
photograph.

Whether this actually helps is an experiment, not an assumption: `p_pattern` defaults to
0 and the effect is measured on both the validation split and ood_check.py before any
checkpoint built with it is shipped.
"""
import numpy as np


def _grating(rng, size):
    pitch = rng.uniform(4.0, 40.0)
    ang = rng.uniform(0.0, 180.0)
    duty = rng.uniform(0.25, 0.75)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    t = np.deg2rad(ang)
    phase = rng.uniform(0, pitch)
    proj = (xx * np.cos(t) + yy * np.sin(t) + phase) % pitch
    return (proj < pitch * duty).astype(np.float32)


def _contacts(rng, size):
    pitch = rng.uniform(8.0, 48.0)
    radius = pitch * rng.uniform(0.15, 0.40)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy = (yy + rng.uniform(0, pitch)) % pitch - pitch / 2
    dx = (xx + rng.uniform(0, pitch)) % pitch - pitch / 2
    return (np.hypot(dy, dx) > radius).astype(np.float32)


def _checker(rng, size):
    cell = int(rng.integers(3, 24))
    yy, xx = np.mgrid[0:size, 0:size]
    return (((yy // cell) + (xx // cell)) % 2).astype(np.float32)


def _steps(rng, size):
    """Piecewise-constant regions with hard edges."""
    img = np.zeros((size, size), np.float32)
    for _ in range(int(rng.integers(2, 6))):
        y0, x0 = rng.integers(0, size, 2)
        h, w = rng.integers(size // 8, size // 2, 2)
        img[y0:y0 + h, x0:x0 + w] = rng.uniform(0, 1)
    return img


def _cross_grating(rng, size):
    """Two gratings superimposed -- array-like structure."""
    a, b = _grating(rng, size), _grating(rng, size)
    return np.maximum(a, b) if rng.random() < 0.5 else a * b


_KINDS = (_grating, _contacts, _checker, _steps, _cross_grating)


def random_pattern(rng, size=256):
    """One random structured image in [0,1], same contract as a GT array.

    Contrast and offset are randomised so the network does not learn that structure
    always means full black-and-white; real inspection images are rarely saturated.
    """
    img = _KINDS[int(rng.integers(len(_KINDS)))](rng, size)
    lo = rng.uniform(0.0, 0.35)
    hi = rng.uniform(lo + 0.25, 1.0)
    img = lo + img * (hi - lo)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _demo():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        p = random_pattern(rng, 64)
        assert p.shape == (64, 64) and p.dtype == np.float32
        assert 0.0 <= p.min() and p.max() <= 1.0, (p.min(), p.max())
        seen.add(round(float(p.std()), 3))
    # If every draw were identical the augmentation would add nothing.
    assert len(seen) > 50, f"patterns are not varied enough: {len(seen)} distinct stds"
    # And they must actually be structured, not flat.
    stds = [random_pattern(rng, 128).std() for _ in range(50)]
    assert np.mean(stds) > 0.05, f"patterns are too flat: mean std {np.mean(stds):.3f}"
    print(f"CHECK: patterns.py self-check passed ({len(seen)} distinct textures, "
          f"mean contrast {np.mean(stds):.3f}, all within [0,1])")


if __name__ == "__main__":
    _demo()
