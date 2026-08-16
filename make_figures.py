"""Generate the figures for the report and slides.

    python make_figures.py --weights weights/model.pt

Produces in results/figures/:
  degradation_analysis.png  -- evidence for the recovered forward model
  training_curve.png        -- loss and validation PSNR against epoch
  restored_best_*.png       -- successful restorations, full resolution
  restored_worst_*.png      -- the failure case KLA explicitly asks for

Success AND failure cases are both required by the KLA brief. The worst case is
picked by PSNR on the validation split rather than chosen by eye, so it is the real
worst rather than a flattering one.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import split_names
from src.degrade import add_noise, area_downsample, fit_noise_levels
from src.metrics import psnr
from src.model import build

OUT = Path("results/figures")


def degradation_analysis(gt_dir, lr_dir, names, n=60):
    """Show that var(residual) rises linearly with pixel^2, and that our simulator
    reproduces it. This is the evidence behind src/degrade.py in one picture."""
    sample = names[:n]
    xs, rs = [], []
    for nm in sample:
        gt, lr = np.load(gt_dir / nm), np.load(lr_dir / nm)
        x = area_downsample(gt)
        xs.append(x.ravel()); rs.append(lr.ravel() - x.ravel())
    x, r = np.concatenate(xs), np.concatenate(rs)

    srng = np.random.default_rng(0)
    sxs, srs = [], []
    for nm in sample:
        gt, lr = np.load(gt_dir / nm), np.load(lr_dir / nm)
        s, g = fit_noise_levels(gt, lr)
        xc = area_downsample(gt)
        sxs.append(xc.ravel()); srs.append(add_noise(xc, s, g, srng).ravel() - xc.ravel())
    sx, sr = np.concatenate(sxs), np.concatenate(srs)

    bins = np.linspace(0, 1, 11)
    mids = (bins[:-1] + bins[1:]) / 2
    real_v = [r[(np.digitize(x, bins) - 1) == b].var() for b in range(10)]
    syn_v = [sr[(np.digitize(sx, bins) - 1) == b].var() for b in range(10)]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(mids ** 2, real_v, "o-", label="real KLA pairs")
    ax[0].plot(mids ** 2, syn_v, "s--", label="our simulator")
    ax[0].set_xlabel("clean pixel value$^2$"); ax[0].set_ylabel("var(residual)")
    ax[0].set_title("Speckle is multiplicative:\nnoise variance $\\propto$ signal$^2$")
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].hist(r, bins=200, density=True, alpha=.6, label="real residual")
    ax[1].hist(sr, bins=200, density=True, alpha=.6, label="simulated residual")
    ax[1].set_xlabel("NoisyLR - area_downsample(GT)"); ax[1].set_ylabel("density")
    ax[1].set_title("Residual distribution, real vs simulated")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "degradation_analysis.png", dpi=150)
    plt.close(fig)
    print("  degradation_analysis.png")


def dataset_sample(gt_dir, n=24, seed=7):
    """What the training data actually IS.

    Worth a figure because it is not what the problem title implies. The KLA
    training set is generic grayscale natural imagery -- books, foliage, pebbles,
    buildings, printed text, animals -- not semiconductor inspection images. That
    reframes the whole task: the hidden test set's "out-of-distribution" half is
    very likely genuine semiconductor content, a domain absent from training. It is
    the reason this solution stays a content-agnostic local restoration operator
    rather than learning priors specific to the training imagery.
    """
    names = sorted(p.name for p in gt_dir.glob("*.npy"))
    rng = np.random.default_rng(seed)
    pick = [names[i] for i in rng.choice(len(names), min(n, len(names)), replace=False)]
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    for ax, nm in zip(axes.ravel(), pick):
        arr = np.load(gt_dir / nm)
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1); ax.axis("off")
        ax.set_title(f"{nm[:-4]}  mean {arr.mean():.2f}", fontsize=7)
    fig.suptitle("Random sample of the KLA training ground truth: generic grayscale "
                 "natural images, not semiconductor inspection data", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "dataset_sample.png", dpi=110)
    plt.close(fig)
    print("  dataset_sample.png")


def training_curve():
    logs = sorted(Path("results").glob("*_log.csv"))
    logs = [p for p in logs if p.stem != "smoke_log"]
    if not logs:
        print("  (no training log yet, skipping training_curve.png)")
        return
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    for p in logs:
        rows = list(csv.DictReader(p.open()))
        if not rows:
            continue
        ep = [int(r["epoch"]) for r in rows]
        ax1.plot(ep, [float(r["train_loss"]) for r in rows], label=f"{p.stem} loss")
        ax2.plot(ep, [float(r["val_psnr"]) for r in rows], "--",
                 label=f"{p.stem} val PSNR")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss (Charbonnier)")
    ax2.set_ylabel("validation PSNR (dB)")
    ax1.set_yscale("log"); ax1.grid(alpha=.3)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "training_curve.png", dpi=150)
    plt.close(fig)
    print("  training_curve.png")


def slide_strip(gt_dir, lr_dir, names, model, device, n=5):
    """A wide, short before/after strip sized for the deck's free space.

    The template leaves roughly 11.1 x 1.1 inches under the metric boxes, so this is
    built at a ~10:1 aspect on purpose rather than shrinking a square figure until the
    text is unreadable.
    """
    picks, seen = [], set()
    for nm in names:
        lr, gt = np.load(lr_dir / nm), np.load(gt_dir / nm)
        if gt.std() < 0.12:          # skip near-flat frames, they show nothing
            continue
        with torch.no_grad():
            pred = model.restore(torch.from_numpy(lr)[None, None].to(device))
        picks.append((lr, pred[0, 0].float().cpu().numpy()))
        seen.add(nm)
        if len(picks) == n:
            break
    fig, axes = plt.subplots(1, 2 * len(picks), figsize=(2 * len(picks) * 1.05, 1.18))
    for i, (lr, pred) in enumerate(picks):
        up = F.interpolate(torch.from_numpy(lr)[None, None].float(), scale_factor=2,
                           mode="nearest")[0, 0].numpy()
        for j, (img, lab) in enumerate(((np.clip(up, 0, 1), "degraded"),
                                        (pred, "restored"))):
            ax = axes[2 * i + j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#c00" if j == 0 else "#0a0"); sp.set_linewidth(1.2)
            ax.set_title(lab, fontsize=6, pad=1.5)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.86, bottom=0.02, wspace=0.06)
    fig.savefig(OUT / "slide_result.png", dpi=220)
    plt.close(fig)
    print("  slide_result.png  (deck strip)")


def triptych(lr, pred, gt, path, title):
    up = F.interpolate(torch.from_numpy(lr)[None, None].float(), scale_factor=2,
                       mode="bicubic", align_corners=False).clamp_(0, 1)[0, 0].numpy()
    panels = [(np.clip(lr, 0, 1), f"degraded input {lr.shape[0]}x{lr.shape[1]}"),
              (up, f"bicubic x2  {psnr(up, gt):.2f} dB"),
              (pred, f"our restoration  {psnr(pred, gt):.2f} dB"),
              (gt, "ground truth")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (img, lab) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(lab, fontsize=10); ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--weights", default="weights/model.pt")
    ap.add_argument("--n_show", type=int, default=2)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    gt_dir, lr_dir = Path(a.data) / "GT", Path(a.data) / "NoisyLR"
    _, val_names = split_names(gt_dir)
    print("Writing figures to", OUT)

    dataset_sample(gt_dir)
    degradation_analysis(gt_dir, lr_dir, val_names)
    training_curve()

    if not Path(a.weights).exists():
        print(f"  (no weights at {a.weights}, skipping restoration figures)")
        return

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ck = torch.load(a.weights, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = build(ck.get("cfg", {})).to(device).eval()
    model.load_state_dict(ck["model"])

    # Score every validation image against BOTH the model and the bicubic baseline.
    #
    # Cases are chosen by GAIN OVER BICUBIC, not by absolute PSNR. Absolute PSNR just
    # finds the darkest, emptiest images -- on a near-black frame everything scores
    # 38 dB and bicubic actually beats us. That makes a flattering-looking figure that
    # demonstrates nothing. Gain over baseline shows what the model actually adds.
    scored = []
    for nm in val_names:
        lr, gt = np.load(lr_dir / nm), np.load(gt_dir / nm)
        with torch.no_grad():
            pred = model.restore(torch.from_numpy(lr)[None, None].to(device))
        pred = pred[0, 0].float().cpu().numpy()
        up = F.interpolate(torch.from_numpy(lr)[None, None].float(), scale_factor=2,
                           mode="bicubic", align_corners=False).clamp_(0, 1)[0, 0].numpy()
        p_model, p_base = psnr(pred, gt), psnr(up, gt)
        scored.append((p_model - p_base, p_model, p_base, nm, lr, pred, gt))

    by_gain = sorted(scored, key=lambda t: t[0])
    for i, row in enumerate(by_gain[-a.n_show:][::-1]):
        g, pm, pb, nm, lr, pred, gt = row
        triptych(lr, pred, gt, OUT / f"restored_best_{i + 1}.png",
                 f"Success case -- {nm}: {pm:.2f} dB vs bicubic {pb:.2f} dB ({g:+.2f} dB)")
        print(f"  restored_best_{i + 1}.png  ({nm}, {pm:.2f} dB, {g:+.2f} vs bicubic)")

    mid = by_gain[len(by_gain) // 2]
    g, pm, pb, nm, lr, pred, gt = mid
    triptych(lr, pred, gt, OUT / "restored_median.png",
             f"Typical case -- {nm}: median of {len(scored)} validation images, "
             f"{pm:.2f} dB vs bicubic {pb:.2f} dB ({g:+.2f} dB)")
    print(f"  restored_median.png  ({nm}, {pm:.2f} dB, {g:+.2f} vs bicubic)  <- representative")

    slide_strip(gt_dir, lr_dir, val_names, model, device)

    # Re-bind to the WORST case. Losing this line once made the "failure case" figure
    # silently show the median image instead -- a figure that claims to be the worst
    # result but is not is worse than no figure at all.
    g, pm, pb, nm, lr, pred, gt = by_gain[0]
    triptych(lr, pred, gt, OUT / "restored_worst_1.png",
             f"Failure case -- {nm}: worst of {len(scored)} by gain over baseline, "
             f"{pm:.2f} dB vs bicubic {pb:.2f} dB ({g:+.2f} dB)")
    print(f"  restored_worst_1.png  ({nm}, {pm:.2f} dB, {g:+.2f} vs bicubic)  <- required failure case")

    gains = np.array([t[0] for t in scored])
    psnrs = np.array([t[1] for t in scored])
    print(f"\nValidation PSNR : worst {psnrs.min():.2f}  median {np.median(psnrs):.2f}  "
          f"best {psnrs.max():.2f} dB")
    print(f"Gain over bicubic: worst {gains.min():+.2f}  median {np.median(gains):+.2f}  "
          f"best {gains.max():+.2f} dB   (beaten by bicubic on "
          f"{(gains < 0).sum()}/{len(gains)} images)")


if __name__ == "__main__":
    main()
