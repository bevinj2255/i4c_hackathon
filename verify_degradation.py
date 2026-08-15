"""Prove src/degrade.py reproduces the real degradation. Aborts if it does not.

This exists because a comment saying "the downsampling is area-mean" does not stop
anyone training against the wrong operator. This script re-derives every constant in
src/degrade.py straight from the provided pairs and fails loudly on disagreement.

Three independent checks:
  1. The downsampling operator: area-mean must leave a smaller residual than either
     subsampling phase. If it does not, our forward model is wrong.
  2. Noise structure: residual variance must rise linearly with pixel^2 (speckle) off
     a small positive intercept (gaussian), and the residual must be spatially white
     (noise applied after downsampling, not before).
  3. Round trip: synthesizing with src/degrade.py and re-fitting must land on the same
     noise statistics as the real data.

    python verify_degradation.py [--n 60]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.degrade import (SIGMA_GAUSS_MEASURED, SIGMA_SPECKLE_MEASURED, add_noise,
                         area_downsample, fit_noise_levels, synthesize)

TOL = 0.15  # 15% relative agreement between real and synthetic noise variances


def fail(msg):
    raise SystemExit(f"ABORT: {msg}")


def ok(msg):
    print(f"CHECK: {msg}")


def pooled_fit(pairs):
    """Fit (speckle_var, gauss_var) over many pairs pooled together."""
    xs, rs = [], []
    for gt, lr in pairs:
        x = area_downsample(gt)
        xs.append(x.ravel())
        rs.append(lr.ravel() - x.ravel())
    x = np.concatenate(xs)
    r = np.concatenate(rs)
    A = np.stack([x ** 2, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, r ** 2, rcond=None)
    return float(coef[0]), float(coef[1]), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    gt_dir, lr_dir = Path(a.data) / "GT", Path(a.data) / "NoisyLR"
    names = sorted(p.name for p in gt_dir.glob("*.npy"))
    if not names:
        fail(f"no data in {gt_dir} -- run prepare_data.py first")
    rng = np.random.default_rng(0)
    sample = [names[i] for i in rng.choice(len(names), size=min(a.n, len(names)), replace=False)]
    pairs = [(np.load(gt_dir / n), np.load(lr_dir / n)) for n in sample]
    print(f"Verifying against {len(pairs)} real pairs from {a.data}\n")

    # --- 1. downsampling operator ------------------------------------------------
    cands = {
        "area-mean (our model)": lambda g: area_downsample(g),
        "subsample [::2]": lambda g: g[::2, ::2],
        "subsample [1::2]": lambda g: g[1::2, 1::2],
    }
    stds = {}
    for label, fn in cands.items():
        r = np.concatenate([(lr - fn(gt)).ravel() for gt, lr in pairs])
        stds[label] = float(r.std())
    best = min(stds, key=stds.get)
    for label, s in stds.items():
        print(f"    residual std, {label:24s} = {s:.5f}")
    if best != "area-mean (our model)":
        fail(f"downsampling operator is NOT area-mean -- {best} fits better. "
             f"src/degrade.py is modelling the wrong operator.")
    ok(f"downsampling is area-mean (residual std {stds[best]:.5f}, "
       f"beats next best by {min(v for k, v in stds.items() if k != best) - stds[best]:.5f})")

    # --- 2. noise structure ------------------------------------------------------
    sv, gv, resid = pooled_fit(pairs)
    if sv <= 0 or gv < 0:
        fail(f"nonsensical noise fit: speckle_var={sv}, gauss_var={gv}")
    ok(f"speckle is multiplicative: sigma_s = {np.sqrt(sv):.4f} (var {sv:.5f})")
    ok(f"additive gaussian: sigma_g = {np.sqrt(gv):.4f} (var {gv:.6f})")

    # linearity of var(r) against x^2 -- the actual evidence for multiplicative noise
    xs = np.concatenate([area_downsample(gt).ravel() for gt, _ in pairs])
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(xs, bins) - 1
    pred, meas = [], []
    for b in range(10):
        m = idx == b
        if m.sum() < 500:
            continue
        mid = ((bins[b] + bins[b + 1]) / 2) ** 2
        pred.append(sv * mid + gv)
        meas.append(resid[m].var())
    corr = float(np.corrcoef(pred, meas)[0, 1])
    if corr < 0.99:
        fail(f"var(residual) vs pixel^2 is not linear (r={corr:.4f}) -- "
             f"the noise is not the speckle+gaussian model we assume")
    ok(f"var(residual) tracks pixel^2 linearly across {len(pred)} bins (r={corr:.5f})")

    # whiteness -- noise applied AFTER downsampling
    lag_h = lag_v = 0.0
    for gt, lr in pairs[:20]:
        r = lr - area_downsample(gt)
        r = (r - r.mean()) / r.std()
        lag_h += float((r[:, :-1] * r[:, 1:]).mean()) / 20
        lag_v += float((r[:-1] * r[1:]).mean()) / 20
    if max(abs(lag_h), abs(lag_v)) > 0.15:
        fail(f"residual is spatially correlated (lag-1 {lag_h:.3f}/{lag_v:.3f}) -- "
             f"noise was applied BEFORE downsampling, our model has the order wrong")
    ok(f"residual is spatially white (lag-1 {lag_h:+.4f} horiz, {lag_v:+.4f} vert) "
       f"=> noise applied after downsampling")

    # --- 3. per-image spread vs the constants in degrade.py ----------------------
    fits = [fit_noise_levels(gt, lr) for gt, lr in pairs]
    ss = np.array([f[0] for f in fits])
    gg = np.array([f[1] for f in fits])
    print(f"    per-image sigma_s: {ss.min():.4f}..{ss.max():.4f} (mean {ss.mean():.4f})")
    print(f"    per-image sigma_g: {gg.min():.4f}..{gg.max():.4f} (mean {gg.mean():.4f})")
    for name, arr, declared in (("sigma_s", ss, SIGMA_SPECKLE_MEASURED),
                                ("sigma_g", gg, SIGMA_GAUSS_MEASURED)):
        lo, hi = float(arr.min()), float(arr.max())
        if lo < declared[0] - 0.05 or hi > declared[1] + 0.05:
            fail(f"{name} range {lo:.4f}..{hi:.4f} has drifted from the declared "
                 f"{declared} in src/degrade.py -- update the constants")
    ok("per-image noise spread matches the ranges declared in src/degrade.py")

    # --- 4. round trip: synthetic must look like real ---------------------------
    srng = np.random.default_rng(1)
    synth = []
    for gt, lr in pairs:
        s, g = fit_noise_levels(gt, lr)
        synth.append((gt, add_noise(area_downsample(gt), s, g, srng)))
    ssv, sgv, _ = pooled_fit(synth)
    for label, real, syn in (("speckle var", sv, ssv), ("gaussian var", gv, sgv)):
        rel = abs(syn - real) / max(real, 1e-9)
        if rel > TOL:
            fail(f"synthetic {label} {syn:.6f} differs from real {real:.6f} "
                 f"by {rel:.1%} (tolerance {TOL:.0%}) -- the simulator is not "
                 f"reproducing the real degradation")
        ok(f"synthetic {label} {syn:.6f} matches real {real:.6f} "
           f"({rel:.1%} apart, tolerance {TOL:.0%})")

    # sanity: the simulator's own sampling range must actually cover the real data
    _, s_drawn, g_drawn = synthesize(pairs[0][0], np.random.default_rng(2))
    ok(f"simulator draws sigma_s={s_drawn:.4f}, sigma_g={g_drawn:.4f} from the "
       f"widened training ranges")

    print("\nPASS: src/degrade.py reproduces the measured degradation.")


if __name__ == "__main__":
    main()
