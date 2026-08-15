"""The forward degradation model, reverse-engineered from the provided pairs.

KLA did not disclose the degradation parameters or the order they were applied in.
We recovered them from the 3200 provided (GT, NoisyLR) pairs:

  1. DOWNSAMPLING is a 2x2 area average (mean of each disjoint 2x2 block).
     Measured against the alternatives on 40 random pairs, residual std was
     0.0862 for area-mean vs 0.0929 for GT[::2,::2] and 0.0942 for GT[1::2,1::2],
     and the fitted additive-noise variance was likewise lowest for area-mean
     (0.00065 vs 0.0018 / 0.0021). Area-mean is the only one that leaves a
     residual consistent with pure noise.

  2. NOISE IS APPLIED AFTER DOWNSAMPLING, not before. The residual at the
     low-resolution grid is spatially white (lag-1 correlation -0.029 horizontal,
     -0.036 vertical). Noise added at 256x256 and then block-averaged would not
     leave this variance structure.

  3. SPECKLE is multiplicative. Regressing residual^2 on pixel^2 over 10 value
     bins gives a straight line through a small positive intercept:

         bin 0.0-0.1  var(r) = 0.000287
         bin 0.5-0.6  var(r) = 0.009181
         bin 0.9-1.0  var(r) = 0.025516

     slope = sigma_speckle^2 = 0.0277, intercept = sigma_gauss^2 = 0.00065.
     Predicted var at the top bin is 0.0277*0.9025 + 0.00065 = 0.0257 against
     0.0255 measured.

  So:  y = x + x * N(0, sigma_s^2) + N(0, sigma_g^2)

Per-image fits over 30 images give sigma_s in [0.099, 0.210] (mean 0.166) and
sigma_g in [0.000, 0.149] (mean 0.028). The sampling ranges below are deliberately
WIDER than measured, because the hidden test set is stated to draw noise levels
that "may vary within a similar range" and half of it is out-of-distribution
content. Training on a wider range than we observed costs a little in-distribution
accuracy and buys robustness where half the marks are.

verify_degradation.py re-derives these numbers from the data and aborts if this
module has drifted away from them.
"""
import numpy as np

# Measured per-image spread, then widened for out-of-distribution robustness.
# The measured range grew when the sample grew (30 pairs said sigma_s topped out at
# 0.210; 60 pairs found 0.245). A first measurement is not a confirmation, so the
# training range carries real margin over what we have actually seen.
SIGMA_SPECKLE_MEASURED = (0.099, 0.245)
SIGMA_GAUSS_MEASURED = (0.000, 0.149)
SIGMA_SPECKLE_RANGE = (0.07, 0.28)
SIGMA_GAUSS_RANGE = (0.00, 0.18)


def area_downsample(hr, factor=2):
    """Average each disjoint factor x factor block. This is the measured operator."""
    h, w = hr.shape[-2:]
    if h % factor or w % factor:
        raise ValueError(f"{hr.shape} not divisible by {factor}")
    return hr.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def add_noise(lr, sigma_s, sigma_g, rng):
    """y = x + x*N(0,sigma_s^2) + N(0,sigma_g^2).

    Not clipped: NoisyLR genuinely runs outside [0,1] in the provided data and
    KLA calls that "a feature not a bug". Clipping here would train the model on
    inputs it will never actually receive.
    """
    speckle = lr * rng.normal(0.0, sigma_s, lr.shape)
    gauss = rng.normal(0.0, sigma_g, lr.shape)
    return (lr + speckle + gauss).astype(np.float32)


def synthesize(gt, rng, factor=2,
               sigma_s_range=SIGMA_SPECKLE_RANGE,
               sigma_g_range=SIGMA_GAUSS_RANGE):
    """Clean high-res image -> degraded low-res image, one fresh random draw.

    Used to generate unlimited training pairs from the 3200 clean GT images, so
    the model never sees the same noise realisation twice.
    """
    sigma_s = rng.uniform(*sigma_s_range)
    sigma_g = rng.uniform(*sigma_g_range)
    return add_noise(area_downsample(gt, factor), sigma_s, sigma_g, rng), sigma_s, sigma_g


def fit_noise_levels(gt, noisy_lr, factor=2):
    """Recover (sigma_speckle, sigma_gauss) from one pair by least squares.

    var(residual | x) = x^2 * sigma_s^2 + sigma_g^2, so regressing the squared
    residual on the squared clean pixel gives slope = sigma_s^2, intercept =
    sigma_g^2. This is the measurement that produced the constants above, kept
    here so verify_degradation.py can re-run it as an independent check.
    """
    x = area_downsample(gt, factor).ravel()
    r = (noisy_lr.ravel() - x)
    A = np.stack([x ** 2, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, r ** 2, rcond=None)
    return float(np.sqrt(max(coef[0], 0.0))), float(np.sqrt(max(coef[1], 0.0)))


def _demo():
    """Round-trip check: synthesize with known sigmas, recover them by fitting."""
    rng = np.random.default_rng(0)
    gt = rng.random((256, 256)).astype(np.float32)
    # A flat random field has little structure; use a smooth one so area-mean matters.
    yy, xx = np.mgrid[0:256, 0:256] / 255.0
    gt = ((np.sin(xx * 9) * np.cos(yy * 7) + 1) / 2).astype(np.float32)

    assert area_downsample(gt).shape == (128, 128)
    assert np.isclose(area_downsample(np.ones((4, 4), np.float32)), 1.0).all()

    lr = area_downsample(gt)
    true_s, true_g = 0.17, 0.03
    noisy = add_noise(lr, true_s, true_g, np.random.default_rng(1))
    got_s, got_g = fit_noise_levels(gt, noisy)
    assert abs(got_s - true_s) < 0.02, f"speckle recovery off: {got_s} vs {true_s}"
    assert abs(got_g - true_g) < 0.02, f"gauss recovery off: {got_g} vs {true_g}"

    noisy2, s, g = synthesize(gt, np.random.default_rng(2))
    assert noisy2.shape == (128, 128) and noisy2.dtype == np.float32
    assert SIGMA_SPECKLE_RANGE[0] <= s <= SIGMA_SPECKLE_RANGE[1]
    assert SIGMA_GAUSS_RANGE[0] <= g <= SIGMA_GAUSS_RANGE[1]
    print(f"CHECK: degrade.py self-check passed "
          f"(recovered sigma_s={got_s:.4f} vs {true_s}, sigma_g={got_g:.4f} vs {true_g})")


if __name__ == "__main__":
    _demo()
