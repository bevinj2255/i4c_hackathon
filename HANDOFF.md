# Handoff — training on a second machine

**Status: LIVE.** Read this if you are picking up training on a different (faster) GPU.

Everything except the trained weights is already done and pushed. What is needed from a
second machine is one thing: **a better checkpoint than the one a 4 GB GTX 1650 can
produce overnight.**

---

## What you need

- An NVIDIA GPU with more than 4 GB of VRAM
- Python 3.10+
- The KLA dataset (`train.zip`, ~919 MB) — the same file, from the official Drive link

## Steps

```bash
git clone https://github.com/bevinj2255/i4c_hackathon.git
cd i4c_hackathon

# Linux: PyPI torch already includes CUDA.
pip install -r requirements.txt
# Windows only, CUDA build must come from the PyTorch index:
#   pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128

# Put train.zip and Test_NoisyLR.zip in the repo root, then:
python prepare_data.py

# 1. Confirm the environment and the data are sound (takes seconds, and it ABORTS on
#    a problem rather than printing a warning you might miss):
python verify_degradation.py
python src/model.py && python -m src.data

# 2. Measure what the GPU can actually do, and pick the config from the measurement:
python train.py --config configs/base.json  --benchmark
python train.py --config configs/large.json --benchmark

# 3. Sanity-check the pipeline before committing hours to it:
python train.py --config configs/large.json --overfit 2

# 4. Train. Use `large` if the benchmark says an epoch takes under ~2 minutes.
python train.py --config configs/large.json
```

Training is **resumable** — if it dies, rerun with
`--resume weights/large_last.pt`. It checkpoints every epoch, so a crash costs one epoch.

## Benchmark AMP before trusting it — it is not a free win

On the GTX 1650 this project was developed on, **fp16 autocast measured 5× SLOWER than
fp32** (4.0 vs 20.1 img/s). The GTX 16-series has no tensor cores, so autocast only adds
conversion overhead and worse cuDNN kernel choices. `channels_last` lost for the same
reason. `configs/base.json` therefore sets `"amp": false`.

**Your GPU almost certainly reverses this.** Anything with tensor cores — T4, RTX 20xx
and up, A100, H100 — should be much faster with `"amp": true`, which is what
`configs/large.json` sets. Confirm it rather than assume:

```bash
python train.py --config configs/large.json --benchmark      # amp true
# then flip "amp" to false in the config and run it again; keep the faster one
```

The `--benchmark` numbers come from short bursts and will overstate sustained throughput
on a thermally-limited laptop GPU. Trust the per-epoch time printed by the real run when
you size the epoch count.

## Which config

| Config | Channels × blocks | Params | Use when |
|---|---|---|---|
| `configs/base.json` | 64 × 16 | 1.37 M | ≤ 4 GB VRAM |
| `configs/large.json` | 96 × 24 | ~4.6 M | > 6 GB VRAM |

Bigger is not automatically better here: KLA scores **end-to-end throughput on an H100**
alongside quality, and their brief warns that unnecessarily large models lose on it. If
`large` gains less than ~0.3 dB over `base`, ship `base`.

## What to send back

Just the checkpoint:

```
weights/large_best.pt        (or base_best.pt)
results/large_log.csv        (the per-epoch training log)
```

Then on either machine:

```bash
cp weights/large_best.pt weights/model.pt
python evaluate.py --weights weights/model.pt          # PSNR / SSIM / LPIPS vs bicubic
python inference.py --input_dir data/test/NoisyLR --output_dir results/restored_test
python make_figures.py --weights weights/model.pt
```

`evaluate.py` **refuses** to accept a checkpoint that does not beat the bicubic
baseline, so a bad run cannot quietly become the submission.

---

## FINALISE — the exact steps to produce a submittable repo

One command rebuilds every deliverable from a checkpoint and **aborts if any of KLA's
eight mandatory pieces is missing**:

```bash
python package_submission.py --checkpoint weights/base_best.pt \
    --team "YourTeam" --members "A (lead), B, C" --college "..." --contact "..."
```

Add `--device cpu` if a training run is still using the GPU. Drop `--skip_tta` to also
measure the 8× self-ensemble.

That produces: `weights/model.pt`, `results/metrics.json`, `results/restored_test/` (397
images), `results/figures/*`, updated README tables, and
`results/<Team>_KLA_PS01.pptx`.

Then, and only then, commit the restored outputs (they are gitignored during
development so each repackage doesn't add another 100 MB to history):

```bash
# remove the results/restored_test/ line from .gitignore first
git add -A && git commit -m "Final submission: restored test outputs and metrics"
git push
```

Last, export the .pptx to PDF and rename it `<Team>_KLA_PS01.pdf` for the portal.

**Before submitting, dry-run the reviewer's path in a clean checkout** — this is the one
that decides whether the submission can be scored at all:

```bash
git clone <repo> /tmp/check && cd /tmp/check
pip install -r requirements.txt
python inference.py --input_dir <some degraded npy dir> --output_dir /tmp/out
```

## Things that will bite you if nobody tells you

1. **Test filenames collide with train filenames.** All 397 test files share names with
   training files, but they are *different images* (pixel correlation ≈ 0.00). Never
   extract them into the same directory. `prepare_data.py` keeps them apart.
2. **Do not clip the input.** NoisyLR genuinely runs outside [0,1]; KLA calls that "a
   feature not a bug". Clipping it throws away information about the speckle.
3. **The output must be clamped to [0,1].** KLA does not clip before scoring. This is
   handled inside `RestoreNet.restore()` — do not bypass it by calling `forward()`.
4. **The `--overfit` gate does not expect near-zero loss.** A convolutional net cannot
   memorise a noisy→clean pair to zero; the loss plateaus around 0.018 and that is
   correct. The gate checks that the loss falls at least 2× and that the model beats
   bicubic on its own training images.
5. **Don't retrain on the test inputs.** Explicitly forbidden by the brief.

## Deadline

**Phase 1 closes 16 August 2026.** Check the portal for the exact cut-off time.
If time runs short, a finished submission with a weaker checkpoint beats an unfinished
one with a better checkpoint — the repo already works end to end, so it can be submitted
with whatever the best checkpoint is at that moment.
