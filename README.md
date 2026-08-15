# AI-Based Restoration of Degraded Semiconductor Inspection Images

**SEMICON India Hackathon 2026 — KLA Problem Statement (PS01)**

Takes a noisy, half-resolution grayscale inspection image and returns a clean image at
full resolution. One network removes speckle noise, removes additive Gaussian noise and
performs 2× super-resolution in a single forward pass.

---

## Quick start — running inference

This is the entry point KLA benchmarks. It needs no source edits and no configuration.

```bash
git clone https://github.com/bevinj2255/i4c_hackathon.git
cd i4c_hackathon
pip install -r requirements.txt

python inference.py --input_dir /path/to/degraded --output_dir /path/to/restored
```

Everything the script needs — architecture, scale factor, weights — is read from
`weights/model.pt`. There is no config file to keep in sync.

Useful flags (all optional):

| Flag | Default | Effect |
|---|---|---|
| `--batch_size N` | 16 | Images per GPU batch |
| `--device cuda\|cpu` | auto | Force a device |
| `--fp32` | off | Disable half precision |
| `--tta` | off | 8× dihedral self-ensemble: slightly better quality, ~8× slower |

### Input / output contract

| | Format |
|---|---|
| **Input** | Directory of `.npy` files. float32, single channel, any spatial size. Values **may lie outside [0,1]** — this is expected and is not clipped on the way in. |
| **Output** | One `.npy` per input, **identical filename**, float32, exactly 2× larger in each dimension, values clamped to **[0,1]**. |

The script verifies all of this after writing and exits non-zero if any output violates
it, so a silent partial run cannot masquerade as a success.

**Why the output is clamped:** ground truth is normalised to [0,1] and KLA does not clip
or renormalise submitted images before scoring. Anything outside the range is pure
avoidable error, so the clamp lives inside `RestoreNet.restore()` where the inference
path cannot skip it.

---

## Reproducing the training

```bash
python prepare_data.py --train_zip train.zip --test_zip Test_NoisyLR.zip --out data
python verify_degradation.py          # confirms the degradation model against real pairs
python train.py --config configs/base.json --overfit 2   # pipeline sanity check
python train.py --config configs/base.json               # the full run
python evaluate.py --weights weights/base_best.pt        # PSNR / SSIM / LPIPS vs baseline
python make_figures.py --weights weights/base_best.pt
```

Training is resumable (`--resume weights/base_last.pt`) and checkpoints every epoch.
Checkpoints and logs are named after the config, so two configurations can never
overwrite each other's results.

`configs/large.json` is the same pipeline at 96 channels / 24 blocks for machines with
more than 4 GB of VRAM.

---

## What the training data actually is

Before modelling anything we looked at the images. **The KLA training ground truth is
generic grayscale natural imagery** — foliage, books, pebbles, buildings, printed text,
animals — not semiconductor inspection images. See
`results/figures/dataset_sample.png`.

That reframes the task, and two design decisions follow from it:

- The hidden test set's stated **out-of-distribution half is most likely genuine
  semiconductor content**, a domain entirely absent from training. So this solution
  deliberately learns a *content-agnostic local restoration operator* — a 72-pixel
  receptive field, trained across a wide randomised noise range — rather than priors
  specific to the training imagery. There is no natural-image prior to exploit that
  would transfer to a wafer image.
- Dihedral augmentation is justified by **equivariance of the degradation**, not by the
  content being orientation-free. Natural photographs plainly have a canonical "up";
  the reason the augmentation is still valid is that per-pixel noise and 2×2 block
  averaging commute with rotations and flips, so a rotated image's ideal restoration is
  exactly the rotated restoration.

## What makes this approach different: the degradation was reverse-engineered

KLA did not disclose how the images were degraded, only that speckle noise, additive
Gaussian noise and downsampling were involved, in an undisclosed order. We recovered the
forward model from the 3200 provided pairs:

1. **Downsampling is a 2×2 area average.** Measured against the alternatives on 60
   pairs: residual std 0.0908 for area-mean, versus 0.1019 and 0.1006 for the two
   subsampling phases. Only area-mean leaves a residual consistent with pure noise.
2. **Noise is applied after downsampling.** The residual on the low-resolution grid is
   spatially white (lag-1 correlation −0.044 horizontal, −0.026 vertical).
3. **Speckle is multiplicative.** Residual variance rises linearly with pixel value²
   (r = 0.993 across 10 value bins); the slope is the speckle variance and the intercept
   is the Gaussian variance.

So the forward model is:

```
x = area_downsample(GT, 2)
y = x + x·N(0, σs²) + N(0, σg²)      σs ≈ 0.10–0.25,  σg ≈ 0.00–0.15
```

**Why this matters.** It lets us manufacture an unlimited supply of correctly-degraded
training pairs from clean images, with fresh noise every epoch and noise levels drawn
from a range deliberately *wider* than the one we measured. Training on the 3200 fixed
pairs alone teaches the network those particular noise realisations; half of KLA's hidden
test set is out-of-distribution content, and that is exactly what this defends against.

`python verify_degradation.py` re-derives every one of these numbers from the data and
**aborts** if the simulator has drifted from the real pairs. See
`results/figures/degradation_analysis.png` — the simulated curve lies on top of the real
one.

---

## Model

Fully-convolutional residual CNN. All convolutions run at the *input* resolution and a
single `PixelShuffle` produces the output, which costs 4× less than working at output
resolution — the throughput axis and the 4 GB training budget point the same way.

| | |
|---|---|
| Parameters | 1,367,553 |
| Body | 16 residual blocks, 64 channels, residual scaling 0.1 |
| Upsampling | one PixelShuffle ×2 at the end |
| Input handling | centred at 0.5, **never clipped** |
| Output handling | clamped to [0,1] |

Two deliberate departures from a stock EDSR:

- **No global skip from the input.** The standard design adds an upsampled copy of the
  input to the output. That works when the input is clean and merely small; ours is
  noisy, so that path would pipe speckle straight into the prediction. The long skip
  runs from the head features instead.
- **Fully convolutional, no hardcoded sizes**, so the same weights restore 128→256 and
  256→512. All released data is ×2; the brief also mentions 512×512 ground truth.

Loss is Charbonnier (smooth L1), with an optional edge term (`edge_weight` in the config)
evaluated as a separate one-variable experiment.

---

## Results

Validation split: 200 images held out by fixed seed, never trained on and never used for
anything but model selection.

<!-- RESULTS-TABLE: filled from results/metrics.json after the final training run -->
_Populated by `python evaluate.py --weights weights/model.pt`; see
`results/metrics.json` for the machine-readable version._

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Bicubic ×2 (baseline) | 23.234 | 0.5395 | 0.4369 |
| RestoreNet (ours) | 26.753 | 0.6537 | 0.3894 |
| RestoreNet + 8× TTA | _pending_ | _pending_ | _pending_ |
| Untrained control | 11.447 | 0.1503 | — |

Figures in `results/figures/`: dataset sample, degradation analysis, training curve,
and restoration examples.

Restoration examples are selected by **gain over the bicubic baseline**, not by absolute
PSNR. Absolute PSNR simply finds the darkest, emptiest frames — on a near-black image
everything scores ~38 dB and bicubic actually beats the model, which makes a
flattering-looking figure that demonstrates nothing. `restored_best_*` are the largest
gains, `restored_median` is the representative case, and `restored_worst_1` is the
required failure case, chosen by measurement rather than by eye.

---

## Repository layout

```
README.md                 this file
requirements.txt          pinned environment
prepare_data.py           unpack the KLA zips into data/
verify_degradation.py     prove the degradation model matches reality (aborts if not)
train.py                  training, resumable; --overfit sanity check; --benchmark
inference.py              THE BENCHMARKED SCRIPT: input dir -> output dir
evaluate.py               PSNR / SSIM / LPIPS against the bicubic baseline
make_figures.py           figures for the report
configs/                  base.json (4 GB GPUs), large.json (bigger GPUs)
src/degrade.py            the recovered forward model + its self-check
src/data.py               dataset, real/synthetic mix, augmentation, fixed split
src/model.py              RestoreNet
src/metrics.py            PSNR / SSIM / LPIPS
weights/model.pt          submitted checkpoint
results/                  metrics, training logs, figures, restored test outputs
```

Every module under `src/` runs standalone as its own self-check:

```bash
python src/degrade.py && python src/model.py && python -m src.data && python src/metrics.py
```

---

## Hardware, runtime and reproducibility

<!-- HARDWARE-TABLE: filled after the final training run and timing measurement -->

| | |
|---|---|
| Training hardware | NVIDIA GeForce GTX 1650 |
| Training time | 5 epochs, 0.3 h wall clock |
| Inference hardware measured | NVIDIA GeForce GTX 1650 |
| End-to-end runtime | see the timing block printed by `inference.py` (disk read, preprocessing, host↔device transfer, model execution, saving) |
| Batch size | 16 |
| Timing method | `time.perf_counter()` with `torch.cuda.synchronize()` around every GPU stage |
| Seed | 0 (set for `random`, `numpy` and `torch`; recorded in the checkpoint) |

## External resources

No external datasets and no pretrained weights were used. The network is trained from
random initialisation on the KLA-provided training data alone, plus synthetic pairs
generated from that same data by `src/degrade.py`.

LPIPS evaluation uses the `lpips` package (BSD-2-Clause), which downloads pretrained
AlexNet features. That is used **for measurement only** and is not part of the model or
the inference path.

## References

- Kumar, T. et al. (2024). *Image Data Augmentation Approaches: A Comprehensive Survey
  and Future Directions.* IEEE Access, 12.
- Zhai, L. et al. (2023). *A Comprehensive Review of Deep Learning-Based Real-World Image
  Restoration.* IEEE Access, 11, 21049–21067.
- Terven, J. et al. (2025). *A Comprehensive Survey of Loss Functions and Metrics in Deep
  Learning.* Artificial Intelligence Review, 58, 195.
- Monga, V. et al. (2021). *Algorithm Unrolling: Interpretable, Efficient Deep Learning
  for Signal and Image Processing.* IEEE Signal Processing Magazine, 38(2), 18–44.
- Lim, B. et al. (2017). *Enhanced Deep Residual Networks for Single Image
  Super-Resolution.* CVPRW. (residual-block and residual-scaling design)
- Shi, W. et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an
  Efficient Sub-Pixel Convolutional Neural Network.* CVPR. (PixelShuffle)
