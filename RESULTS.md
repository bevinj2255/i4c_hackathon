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

### 1.1 What the images actually are — checked, not assumed

**The training ground truth is generic grayscale natural imagery, not semiconductor
inspection data.** A random sample of 24 (`results/figures/dataset_sample.png`) shows
foliage, books, water, pebbles, buildings, printed text, animals, wood grain.

| Statistic over 600 GT images | Value |
|---|---|
| mean brightness | 0.430 (range 0.039 – 0.932) |
| contrast (std) | 0.184 (range 0.022 – 0.363) |
| very dark images (mean < 0.1) | 9 of 600, 2% |

Three consequences, all of which changed decisions:

1. The hidden test set's stated out-of-distribution half is most likely **genuine
   semiconductor content** — a domain absent from training entirely. This is the
   argument for keeping the model a content-agnostic *local* operator (72 px receptive
   field, wide randomised noise range) instead of learning priors tied to natural
   images, which would not transfer to a wafer image.
2. The dihedral augmentation justification originally written down ("isotropic
   semiconductor textures with no canonical up") was **factually wrong** — natural
   photographs have a canonical up. The augmentation is still valid, but because the
   degradation and its inverse are *equivariant* under the dihedral group, which is a
   different argument. Corrected in `src/data.py`, README and the deck.
3. Figure selection by absolute PSNR is misleading here. The 2% of near-black frames
   score ~38 dB for any method, and bicubic beats the model on them (39.34 vs 37.80 dB
   on the top-PSNR image). Selection changed to **gain over the bicubic baseline**.

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

### 2.5 The noise spectrum — matching variance was not enough

Variance agreement (below, 0.0%) looked like proof the simulator was right. It was not.
Comparing **radially-averaged power spectra** of real vs simulated degraded images
exposed a gap that every variance test had passed straight through:

| | real / simulated high-frequency power |
|---|---|
| white noise simulator | **1.1585** |

Real data carried ~16% more high-frequency power. Two noise fields can share a variance
and have completely different spectra.

**Control run first, before believing it.** The same statistic measured on synthetic data
whose noise is white *by construction*:

| | lag-1 h | lag-1 v |
|---|---|---|
| real residual | −0.0445 | −0.0603 |
| synthetic, known-white | −0.0032 | −0.0002 |

The control is clean, so the blueness is a property of the real noise, not an artefact of
how it was measured.

**Mechanism ruled out.** Injecting noise before a downsample reproduces the sign but
badly overshoots the magnitude:

| Where noise enters | lag-1 h | lag-1 v |
|---|---|---|
| after area-downsample (our model) | −0.0023 | +0.0007 |
| before bicubic downsample | −0.1533 | −0.1551 |
| before bicubic+antialias downsample | +0.1593 | +0.1569 |
| **real data** | **−0.0445** | **−0.0603** |

Six downsampling kernels were also tested directly against the real pairs (area, bicubic
±antialias, bilinear ±antialias, subsample). All left the same negative correlation, so
the kernel is not the cause; area-mean and bicubic-no-antialias tie within 0.3% on
residual std, and area-mean was kept.

**Resolution.** Rather than model a mechanism we could not identify, the measured
statistic is matched directly with a separable 3-tap `[-a, 1, -a]` filter on the noise
fields, for which lag-1 = −2a/(1+2a²). Variance is renormalised so `a` moves only the
spectrum.

| a | lag-1 h | lag-1 v | std ratio |
|---|---|---|---|
| 0.0000 | −0.0027 | −0.0017 | 1.0000 |
| **0.0225** | **−0.0456** | **−0.0433** | 0.9992 |
| 0.0450 | −0.0902 | −0.0914 | 0.9988 |

Result: high-frequency power ratio **1.1585 → 1.0714**. The residual 7% is anisotropy —
real noise is bluer vertically than horizontally — which an isotropic filter cannot
represent. `COLOUR_RANGE = (0.0, 0.045)` brackets the measured value rather than pinning
it, on the same reasoning as the widened σ ranges.

`verify_degradation.py` now checks the autocorrelation as well as the variance, so this
cannot silently regress.

### 2.6 Round-trip validation

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

### 3.1 Receptive field — why training patches were dropped

Measured empirically (which input pixels carry nonzero gradient to one output pixel,
with all-positive weights so no path cancels):

| Blocks | Measured receptive field |
|---|---|
| 8 | 40 input px |
| **16 (base)** | **72 input px** |
| 24 (large) | 104 input px |

The original plan trained on random 64×64 crops. **The receptive field is larger than
that crop.** Every output pixel's context would have been clipped by the patch boundary
and filled with zero padding — padding the network never encounters at inference, where
images are a full 128×128. That is a train/test mismatch built straight into the data
loader.

Changed to `patch: 0` (whole 128×128 image). Both configs' receptive fields fit inside
128 px. Cropping bought nothing anyway: augmentation diversity already comes from the
8 dihedral transforms and a fresh noise draw every epoch.

Alignment re-verified on the real (noisy) pairs, where an exact match is impossible:
zero-shift correlation between `area_downsample(GT)` and `NoisyLR` beats all 8
one-pixel-shifted alternatives by ≥ 0.0226, with and without augmentation.

### 3.2 The overfit gate, and why its criterion changed

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

## 4.2 Throughput sweep — fp16 was 5× SLOWER than fp32

Measured on the GTX 1650, real training steps (forward + backward + optimiser), full
128×128 images, 15 s per point after cuDNN warm-up.

| Configuration | img/s | Peak GiB | min/epoch |
|---|---|---|---|
| 64ch/16blk, batch 8, **AMP fp16** | 4.0 | 2.60 | 12.4 |
| 64ch/16blk, batch 8, **fp32** | **20.1** | 2.34 | **2.5** |
| 64ch/16blk, batch 8, fp32 + channels_last | 15.6 | 2.40 | 3.2 |

**AMP fp16 is 5× slower than fp32 on this card.** The GTX 16-series (Turing TU117) has
no tensor cores, so autocast buys nothing and costs conversion overhead plus worse cuDNN
kernel selection. `channels_last` also loses, for the same reason — it is a layout for
tensor cores that aren't there.

This is card-specific and must be re-measured elsewhere: on a T4/A100/H100 `amp: true`
will very likely win. `configs/large.json` keeps AMP on for that reason, with a note to
benchmark first.

Had this not been measured, the overnight run would have completed **29 epochs instead
of 110**.

### Capacity vs epochs (all fp32)

| Configuration | img/s | Peak GiB | Epochs in 5.5 h |
|---|---|---|---|
| **64ch/16blk, batch 12** | **20.1** | **2.55** | **~133** |
| 64ch/16blk, batch 16 | 20.6 | 2.90 | 136 |
| 64ch/24blk, batch 8 | 14.0 | 2.12 | 92 |
| 96ch/16blk, batch 8 | 10.0 | 2.65 | 66 |
| 96ch/24blk, batch 8 | 7.1 | 3.16 | 47 |
| 128ch/16blk, batch 8 | 6.3 | 2.94 | 42 |

Chose 64ch/16blk at batch 12: batch 16 is marginally faster but 2.90 GiB of 4.00 GiB is
uncomfortable headroom on a card that is also driving the display. Larger models cost
more epochs than the extra capacity is worth in a single overnight run, and a leaner
model also scores better on KLA's throughput axis.

**Synthetic benchmark 149 s/epoch vs real 191 s/epoch.** The gap is data loading and the
200-image validation pass, neither of which the synthetic benchmark includes. Run length
was sized from the real number.

## 5. Training runs

_Populated as runs complete. Each row records config, hardware, epochs, wall clock and
the validation metric it produced._

| Run | Config | Params | Hardware | Epochs | Wall clock | Best val PSNR |
|---|---|---|---|---|---|---|
| base | 64ch/16blk, fp32, batch 12, full 128×128 images, p_synth 0.5 | 1,367,553 | GTX 1650 4 GiB | 110 | 5.40 h | **28.533 dB** (epoch 108) |

Progression: 26.35 dB (epoch 1) → 27.90 (10) → 28.13 (20) → 28.25 (40) → 28.43 (60) →
28.50 (80) → 28.53 (110). Cosine annealed over exactly the run, as sized from the
measured epoch time.

## 5.1 Out-of-distribution probe — generalisation improves with training

15 synthetic semiconductor-like patterns (gratings at 3 pitches × 3 angles, contact
arrays, checkerboards, step edges, and a periodic field with a missing contact plus a
bridging defect), degraded with the *measured* forward model so only the content is out
of distribution. None of this structure appears in training, which is natural
photography.

| | epoch 1 | epoch 60 |
|---|---|---|
| mean PSNR gain vs bicubic | −0.85 dB | **+2.10 dB** |
| patterns where bicubic wins | 10 / 15 | **3 / 15** |
| step edges | +7.18 dB | **+11.70 dB**, SSIM 0.296 → 0.969 |
| defect array | −0.05 dB | **+2.04 dB**, SSIM 0.628 → 0.802 |
| grating pitch 16 (best) | +2.03 dB | **+4.92 dB** |
| grating pitch 6, checker 4 (worst) | −5.9 dB | −1.0 dB |

**The finding that matters:** trained only on natural photographs, the network becomes
good at structure it has never seen. That is evidence it is learning a generic local
restoration operator rather than natural-image priors — which is exactly the property
the hidden test set's out-of-distribution half demands.

The residual losses sit at pitch 6 and a 4-pixel checkerboard. After a 2×2 area
downsample those land essentially on the box filter's null: a 6 px pitch at 256 becomes
3 px at 128. The contrast is destroyed, not merely corrupted, so no method recovers it
and bicubic scores better only by passing through noise that happens to preserve
contrast. This is an information-theoretic floor and is reported as such rather than
hidden.

**Consequence for the planned experiment.** `configs/finetune_ood.json` was written to
mix synthetic structure into training on the assumption that OOD would be the weak
point. This measurement says it largely is not. The experiment is still worth running
with spare GPU time, but the bar stands: it ships only if it improves *both* the
in-distribution validation split and this probe.

## 5.2 Final model: the large network plus a perceptual fine-tune

A second machine (RTX 4050) trained the 96ch/24blk configuration for 147 epochs / 6.11 h,
reaching 29.016 dB — 0.48 dB above our 1.37M model. But it used Charbonnier alone, so its
LPIPS was 0.3099 against our 0.2158. Better on two axes, clearly worse on the third.

Fine-tuned it with the LPIPS term for 25 epochs (3.4 h on the GTX 1650, batch 4), then
swept the interpolation against the original checkpoint:

| Candidate | PSNR | SSIM | LPIPS ↓ |
|---|---|---|---|
| previous shipped model (1.37M) | 28.378 | 0.7477 | 0.2158 |
| **large + LPIPS fine-tune — SHIPPED** | **28.676** | **0.7543** | **0.1811** |
| 25% large + 75% fine-tune | 28.814 | 0.7592 | 0.1996 |
| 50/50 | 28.919 | 0.7624 | 0.2303 |
| 75% large + 25% fine-tune | 28.988 | 0.7641 | 0.2709 |
| large_best (Charbonnier only) | 29.016 | 0.7643 | 0.3099 |

**The pure fine-tune dominates the previous model on all three metrics simultaneously** —
the first candidate to do so rather than trade. It also has the best worst-axis fraction
of any point on the curve (0.94), so no interpolation was needed.

Costs, recorded rather than buried:

| | previous (1.37M) | shipped (4.4M) |
|---|---|---|
| OOD probe mean | **+2.56 dB (1/15 losses)** | +2.32 dB (2/15) |
| inference, fp32, GTX 1650 | **31.5 ms/image** | 101.4 ms/image |

The OOD regression is small and rests on 15 synthetic patterns — weak evidence against
three metrics measured on 200 real images. The 3.2x slowdown is real, but KLA frames
throughput coarsely ("10 minutes versus 10 seconds") and 101 ms on a GTX 1650 — about the
slowest CUDA card in service — becomes single-digit milliseconds on the H100 they
benchmark on. Quality on real data was judged to outweigh both.

TTA re-tested on this checkpoint and rejected for a third time: 28.807 / 0.7594 but LPIPS
0.1921 against 0.1811, at 8x the cost. Three independent checkpoints, same verdict.

Still beats bicubic on 200/200 validation images.

## 6. Final metrics

200-image held-out split, never trained on, sole basis for model selection.

| Method | PSNR (dB) | SSIM | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×2 (baseline) | 23.234 | 0.5395 | 0.4369 |
| **RestoreNet (1.37 M params)** | **28.533** | **0.7514** | **0.3297** |
| RestoreNet + 8× TTA | 28.569 | 0.7525 | 0.3337 |
| Untrained control (same architecture) | 10.387 | 0.1022 | — |

Gain over baseline: **+5.299 dB PSNR, +0.2119 SSIM, 0.1072 better LPIPS**.
**Beaten by bicubic on 0 of 200 images** — worst gain +0.55 dB, median +4.65, best +18.71.

The untrained control at 10.387 dB confirms the metrics are measuring training rather
than architecture alone.

### 6.0 Loss study: closing the perceptual gap

Another team's public results (320-image split) showed where we were weak. Compared as
*gain over each team's own bicubic baseline*, which is the only fair comparison when the
splits differ — theirs scored bicubic at 22.73 dB, ours at 23.234 on the same algorithm:

| Gain over bicubic | Them | Us (Charbonnier only) |
|---|---|---|
| PSNR | +4.44 dB | **+5.30 dB** |
| SSIM | +0.1787 | **+0.2119** |
| LPIPS | **0.1943** | 0.1072 |

Their loss was L1 + SSIM + LPIPS; ours was Charbonnier alone. Charbonnier minimises
pixel error, and the cheapest way to do that is to smooth away low-contrast texture —
visible in our median figure, and precisely what LPIPS punishes. A named, fixable cause.

Fine-tuned the trained model for 25 epochs with a frozen-AlexNet LPIPS term added
(weight 0.10, lr 5e-5). Then, because both models descend from the same parent and so
occupy the same loss basin, interpolated their weights to trace the trade-off:

| Model | PSNR | SSIM | LPIPS ↓ | OOD mean | OOD losses |
|---|---|---|---|---|---|
| base (Charbonnier only) | **28.533** | **0.7514** | 0.3297 | +2.56 dB | 1/15 |
| avg50 (50% base) | 28.461 | 0.7508 | 0.2462 | — | — |
| **avg25 (25% base) — SHIPPED** | 28.378 | 0.7477 | 0.2158 | **+2.66 dB** | **1/15** |
| perceptual (0% base) | 28.265 | 0.7429 | **0.1980** | +2.60 dB | 3/15 |

Expressed as a fraction of the best candidate on each axis:

| Model | PSNR | SSIM | LPIPS | worst axis |
|---|---|---|---|---|
| base | 1.00 | 1.00 | 0.45 | 0.45 |
| avg50 | 0.99 | 1.00 | 0.80 | 0.80 |
| **avg25** | 0.97 | 0.98 | 0.93 | **0.93** |
| perceptual | 0.95 | 0.96 | 1.00 | 0.95 |

**Chose avg25.** It is the *best of every candidate on out-of-distribution content*
(+2.66 dB, losing on only 1 of 15 patterns, where the pure perceptual model loses on 3),
sits within 2–3% of the best on PSNR and SSIM, and captures 93% of the available LPIPS
improvement. Against the competitor it now leads on all three: +5.145 dB, +0.2083 SSIM,
0.2211 LPIPS.

**Deviation from the pre-registered rule, stated plainly.** `configs/perceptual.json`
said "ship only if PSNR drops by less than ~0.15 dB". avg25 drops 0.155 dB — 0.005 over.
The rule was written before the size of the LPIPS gain was known; a 34% LPIPS improvement
for 2.9% of the PSNR gain is a trade the rule did not anticipate. Recording the override
rather than quietly moving the threshold.

Weight averaging cost nothing: it needed no additional training, only two checkpoints
that already existed.

### 6.1 Test-time augmentation — measured, and rejected

| | PSNR | SSIM | LPIPS ↓ | cost |
|---|---|---|---|---|
| plain (base) | 28.533 | 0.7514 | **0.3297** | 1× |
| 8× TTA (base) | **28.569** | **0.7525** | 0.3337 | 8× |
| plain (shipped avg25) | 28.378 | 0.7477 | **0.2158** | 1× |
| 8× TTA (shipped avg25) | **28.449** | **0.7511** | 0.2257 | 8× |

Tested twice, on two different checkpoints, with the same verdict both times: TTA lifts
PSNR and SSIM slightly and makes LPIPS *worse*, for 8× the compute.

TTA buys +0.036 dB and +0.0011 SSIM but makes **LPIPS worse**, for eight times the
compute, on a task that is also scored on throughput. Kept available behind `--tta`,
not enabled. A negative result worth recording: the obvious upgrade is not one here.

### 6.2 Inference throughput

397 test images, batch 16, GTX 1650, end-to-end (disk read → preprocess → transfer →
model → transfer → save):

| Precision | ms/image | images/s | model execution |
|---|---|---|---|
| fp32 (`--fp32`) | **31.51** | **31.7** | 9.86 s |
| fp16 (default) | 113.30 | 8.8 | 42.95 s |

Same 5× fp16 penalty as in training, same cause: no tensor cores. fp16 remains the
default because KLA benchmarks on an H100, where it is the correct choice; `--fp32` is
documented for hardware without tensor cores.

Non-model stages total 1.07 s of the 12.51 s fp32 run — 8.5%. Worth knowing, since KLA
counts them and a model fast enough would become I/O bound.
