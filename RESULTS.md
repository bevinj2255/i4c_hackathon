# RESULTS — every measurement, with its conditions

**Status: LIVE.** Newest entries at the bottom of each section.

Rule for this file: a number without its conditions is not a result. Every entry records
what was measured, on what, and how. "Measured", "estimated" and "read from the tool" are
three different words.

---

## 1. Dataset facts

| Fact | Value | How established |
|---|---|---|
| Train pairs | 3200 | `.npy` headers read for **all 3200** files, not sampled |
| GT shape / dtype | (256, 256) float32, all 3200 | header scan |
| NoisyLR shape / dtype | (128, 128) float32, all 3200 | header scan |
| GT value range | exactly [0, 1] | min/max over sampled arrays |
| NoisyLR value range | ≈ [−0.06, 1.66] | min/max over sampled arrays |
| Released test inputs | 397 files, **all** (128, 128) float32 | every test array shape-counted |
| Test ground truth | not provided | KLA retains it for scoring |

**Trap (measured, not assumed):** all 397 test filenames also exist in the training set,
but they are different images — pixel correlation between same-named train/test files is
−0.006, 0.003, 0.011 on the three checked, and none are byte-identical. The numbering
spaces simply overlap (test IDs 0–399, train IDs 0–3199).

---

## 2. The degradation forward model

Recovered from the provided pairs; not supplied by KLA.

### 2.1 Which downsampling operator

Residual std of `NoisyLR − downsample(GT)` over 60 random pairs. Lower means the
residual is closer to pure noise, i.e. the operator is right.

| Operator | Residual std | Fitted additive-noise var |
|---|---|---|
| **2×2 area mean** | **0.0908** | **0.00065** |
| subsample `[::2, ::2]` | 0.1019 | 0.0018 |
| subsample `[1::2, 1::2]` | 0.1006 | 0.0021 |

Conditions: 60 pairs drawn with seed 0 from `data/train`, pooled over all pixels.
Conclusion: area-mean, by a clear margin on both statistics.

### 2.2 Noise structure

Regressing squared residual on squared clean pixel, pooled over 60 pairs:

- slope = speckle variance = **0.02553** → σ_s = 0.1598
- intercept = Gaussian variance = **0.001338** → σ_g = 0.0366
- linearity of var(residual) against pixel² across 10 bins: **r = 0.99314**

Binned evidence (earlier 40-pair run, same conclusion):

| pixel bin | var(residual) |
|---|---|
| 0.0–0.1 | 0.000287 |
| 0.5–0.6 | 0.009181 |
| 0.9–1.0 | 0.025516 |

Predicted at the top bin: 0.0277 × 0.9025 + 0.00065 = 0.0257 against 0.0255 measured.

### 2.3 Order of operations

Lag-1 autocorrelation of the residual on the low-resolution grid, averaged over 20 pairs:
**−0.0436 horizontal, −0.0264 vertical**. Effectively white, so noise was applied
**after** downsampling. Noise added at 256×256 and then block-averaged would not leave
this structure.

### 2.4 Per-image noise spread

| | 30 pairs | 60 pairs |
|---|---|---|
| σ_speckle | 0.099 – 0.210 (mean 0.166) | 0.1147 – 0.2449 (mean 0.1673) |
| σ_gauss | 0.000 – 0.149 (mean 0.028) | 0.0000 – 0.0825 (mean 0.0295) |

**The range grew when the sample grew.** A first measurement is not a confirmation. The
training sampling ranges therefore carry margin over what was observed: σ_s ∈ [0.07,
0.28], σ_g ∈ [0.00, 0.18].

### 2.5 Round-trip validation

Synthesising with `src/degrade.py` at the per-image fitted σ and re-fitting:

| | real | synthetic | agreement |
|---|---|---|---|
| speckle variance | 0.025532 | 0.025520 | **0.0%** |
| Gaussian variance | 0.001338 | 0.001356 | **1.3%** |

Tolerance is 15%; `verify_degradation.py` aborts outside it. See
`results/figures/degradation_analysis.png` — the simulated curve lies on the real one.

---

## 3. Pipeline checks (all currently passing)

| Check | What it proves | Result |
|---|---|---|
| `src/degrade.py` | synthesised noise can be recovered by fitting | σ_s 0.1747 vs 0.17, σ_g 0.0215 vs 0.03 |
| `src/model.py` | ×2 at three input sizes; `restore()` clamps; zeroed model goes flat | pass, 1,367,553 params |
| `python -m src.data` | crops stay aligned under all 8 augmentations; split deterministic | pass, max misalignment < 1e-5 |
| `src/metrics.py` | PSNR and SSIM both fall when noise rises | pass |
| dihedral TTA | `undihedral(dihedral(x,k),k) == x` for all 8 k, all 8 distinct | pass, on non-square input |
| `inference.py` | end-to-end, output contract enforced | pass on 20 test images (CPU) |

### 3.1 The overfit gate, and why its criterion changed

First attempt asserted training loss < 0.01 on a fixed pair. **It failed, twice, on a
healthy pipeline.**

| Setup | Final loss | Note |
|---|---|---|
| 16ch/2blk, random 32px crops, 40 epochs | 0.02231 | input differs every step |
| 32ch/4blk, fixed whole image, 60 epochs | 0.01778 | plateaued, flat to 4 decimals |

Diagnosis: a fully-convolutional network is a *local* operator, so the same noisy
neighbourhood maps to different clean values in different places. A noisy→clean pair
cannot be memorised to zero at any capacity. The threshold was testing for something
physically unavailable.

Replaced with two criteria a real bug does trip: loss must fall ≥2×, and the model must
beat bicubic by ≥1 dB on its own training images.

**Mutation test of the new gate** — target swapped to a different image:

| | healthy | mutated |
|---|---|---|
| loss fall | 4.5× | 1.59× |
| model vs bicubic | 27.81 vs 26.36 dB | **12.10** vs 26.36 dB |
| gate verdict | pass | **ABORT** |

The gate fails for the reason it exists. Before this, it was a green light wired to
nothing.

---

## 4. Environment

| Item | Value | How |
|---|---|---|
| GPU | NVIDIA GeForce GTX 1650, 4 GB | `nvidia-smi` |
| Driver / CUDA | 596.49 / 13.2 | `nvidia-smi` |
| PyTorch build chosen | 2.9.1+cu128 | cu130 offers 2.12.1 but CUDA 13 dropped older architectures; cu128 is certain to support Turing (sm_75) |

### 4.1 The pip incident (kept as a negative result)

`pip install torch==2.9.1 --index-url .../cu128` ran 79 minutes and installed nothing.

| Measurement | Value |
|---|---|
| bytes written by the pip process | 4.29 GB |
| bytes landed in pip cache | 233 MB, unchanged for 40 min |
| bytes landed in site-packages | 0 |
| partial download files in TEMP | none, none dated today |
| direct download rate, measured separately | 0.46 MiB/s (while competing with pip) |

Conclusion: repeated failed downloads, each discarded and restarted from zero.
Replaced with a resumable downloader using HTTP Range — **measured 2.8 MiB/s, ~5×
faster**, and a dropped connection costs only the bytes in flight.

Lesson worth keeping: "it is stuck" was checked with four cheap decisive measurements
(cache growth, site-packages growth, process I/O counters, partial-file search) before
anything was killed. The I/O counters alone were ambiguous — write was climbing, which
looked like progress — and it took the missing partial file to settle it.

---

## 5. Training runs

_Populated as runs complete. Each row records config, hardware, epochs, wall clock and
the validation metric it produced._

| Run | Config | Params | Hardware | Epochs | Wall clock | Best val PSNR |
|---|---|---|---|---|---|---|
| _pending_ | | | | | | |

## 6. Final metrics

_Populated by `evaluate.py` on the 200-image held-out split._

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| _pending_ | | | |
