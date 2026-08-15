"""Out-of-distribution check on synthetic semiconductor-like patterns.

    python ood_check.py --weights weights/model.pt

Why this exists: the KLA training set turned out to be generic grayscale natural
photography (see results/figures/dataset_sample.png), while the task is framed around
semiconductor inspection and the hidden test set is stated to include out-of-distribution
content. The most likely form that takes is genuine wafer imagery -- a domain the model
never sees in training.

We cannot obtain KLA's hidden data, but we can build the kind of structure that domain
is made of and measure honestly against it: dense line/space gratings, contact-hole
arrays, checkerboards, sharp step edges, and a periodic field containing a small defect
(the thing inspection actually looks for).

These are SYNTHETIC and are not claimed to be real wafer images. They are a
generalisation probe: a model that has merely memorised natural-image statistics will
fall apart on periodic high-contrast structure, and this says by how much.

Degradation uses exactly the forward model recovered in src/degrade.py, so the noise is
the measured one and only the CONTENT is out of distribution.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.degrade import COLOUR_MEASURED, add_noise, area_downsample
from src.metrics import psnr, ssim
from src.model import build

SIZE = 256


def _norm(a):
    a = a.astype(np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)


def line_space(pitch, angle_deg, duty=0.5):
    """Dense grating -- the single most common structure on a wafer."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    t = np.deg2rad(angle_deg)
    proj = xx * np.cos(t) + yy * np.sin(t)
    return _norm(((proj % pitch) < pitch * duty).astype(np.float32))


def contact_array(pitch, radius):
    """Periodic contact/via holes."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    dy = (yy % pitch) - pitch / 2
    dx = (xx % pitch) - pitch / 2
    return _norm((np.hypot(dy, dx) > radius).astype(np.float32))


def checkerboard(cell):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return _norm((((yy // cell) + (xx // cell)) % 2).astype(np.float32))


def step_edges():
    """Large flat regions separated by hard edges -- tests ringing and edge fidelity."""
    img = np.zeros((SIZE, SIZE), np.float32)
    img[:, SIZE // 3:] = 0.5
    img[:, 2 * SIZE // 3:] = 1.0
    img[SIZE // 2:, :] = 1.0 - img[SIZE // 2:, :]
    return img


def defect_in_array(pitch, radius):
    """A periodic field with one contact missing and one bridge added.

    This is the case that actually matters for inspection: the restoration must not
    erase the anomaly, and must not invent one.
    """
    img = contact_array(pitch, radius)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cy = cx = SIZE // 2 - (SIZE // 2) % pitch + pitch / 2
    img[np.hypot(yy - cy, xx - cx) <= radius] = 1.0            # missing contact
    y0 = int(cy + pitch); x0 = int(cx)
    img[y0 - 2:y0 + 2, x0:x0 + int(pitch)] = 0.0               # bridging defect
    return img


def patterns():
    p = {}
    for pitch in (6, 10, 16):
        for ang in (0, 45, 90):
            p[f"grating_p{pitch}_a{ang}"] = line_space(pitch, ang)
    for pitch, r in ((12, 3), (20, 6)):
        p[f"contacts_p{pitch}"] = contact_array(pitch, r)
    p["checker_4"] = checkerboard(4)
    p["checker_8"] = checkerboard(8)
    p["step_edges"] = step_edges()
    p["defect_array"] = defect_in_array(20, 6)
    return p


def bicubic(lr):
    t = torch.from_numpy(lr)[None, None].float()
    return torch.nn.functional.interpolate(
        t, scale_factor=2, mode="bicubic", align_corners=False
    ).clamp_(0, 1)[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/model.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sigma_s", type=float, default=0.166, help="measured mean speckle")
    ap.add_argument("--sigma_g", type=float, default=0.030, help="measured mean gaussian")
    ap.add_argument("--out", default="results/ood_metrics.json")
    ap.add_argument("--figure", default="results/figures/ood_patterns.png")
    a = ap.parse_args()

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ck = torch.load(a.weights, map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    model = build(ck.get("cfg", {})).to(device).eval()
    model.load_state_dict(ck["model"])

    rng = np.random.default_rng(0)
    rows, panels = {}, []
    for name, gt in patterns().items():
        lr = add_noise(area_downsample(gt), a.sigma_s, a.sigma_g, rng, COLOUR_MEASURED)
        with torch.no_grad():
            pred = model.restore(torch.from_numpy(lr)[None, None].float().to(device))
        pred = pred[0, 0].float().cpu().numpy()
        up = bicubic(lr)
        rows[name] = {
            "psnr": psnr(pred, gt), "ssim": ssim(pred, gt),
            "psnr_bicubic": psnr(up, gt), "ssim_bicubic": ssim(up, gt),
        }
        rows[name]["psnr_gain"] = rows[name]["psnr"] - rows[name]["psnr_bicubic"]
        panels.append((name, lr, up, pred, gt))

    print(f"Out-of-distribution probe: {len(rows)} synthetic semiconductor-like patterns")
    print(f"degraded with the measured forward model "
          f"(sigma_s={a.sigma_s}, sigma_g={a.sigma_g}, colour={COLOUR_MEASURED})\n")
    print(f"{'pattern':<20}{'bicubic dB':>12}{'ours dB':>10}{'gain':>8}"
          f"{'bicubic SSIM':>14}{'our SSIM':>10}")
    print("-" * 74)
    for k, v in rows.items():
        print(f"{k:<20}{v['psnr_bicubic']:>12.2f}{v['psnr']:>10.2f}"
              f"{v['psnr_gain']:>+8.2f}{v['ssim_bicubic']:>14.4f}{v['ssim']:>10.4f}")
    gains = np.array([v["psnr_gain"] for v in rows.values()])
    print(f"\nmean gain over bicubic on unseen structure: {gains.mean():+.2f} dB "
          f"(worst {gains.min():+.2f}, best {gains.max():+.2f}); "
          f"bicubic wins on {(gains < 0).sum()}/{len(gains)}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"sigma_s": a.sigma_s, "sigma_g": a.sigma_g, "patterns": rows,
         "mean_psnr_gain": float(gains.mean())}, indent=2))
    print(f"Written to {a.out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = panels[:6]
    fig, axes = plt.subplots(len(show), 4, figsize=(11, 2.6 * len(show)))
    for r, (name, lr, up, pred, gt) in enumerate(show):
        for c, (img, lab) in enumerate([
                (np.clip(lr, 0, 1), "degraded 128"), (up, "bicubic"),
                (pred, "ours"), (gt, "ground truth")]):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(lab, fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=8)
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
    fig.suptitle("Generalisation probe: synthetic semiconductor-like structure, "
                 "absent from training", fontsize=12)
    fig.tight_layout(); fig.savefig(a.figure, dpi=130)
    print(f"Written to {a.figure}")


if __name__ == "__main__":
    main()
