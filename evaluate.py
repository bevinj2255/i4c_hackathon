"""Measure PSNR / SSIM / LPIPS on the held-out validation split.

    python evaluate.py                          # baseline only
    python evaluate.py --weights weights/model.pt
    python evaluate.py --weights weights/model.pt --tta

The 200 validation images are fixed by seed in src/data.py and are never trained on.
Every number in the report and the slides comes from this script.

The baseline is bicubic x2 of the degraded input, clamped to [0,1] -- zero parameters,
and the thing our model has to beat. KLA requires at least one baseline comparison.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import split_names
from src.metrics import summarise
from src.model import build


def bicubic_upscale(lr, scale=2):
    t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False)
    return up.squeeze().clamp_(0.0, 1.0).numpy().astype(np.float32)


def dihedral_t(t, k):
    if k & 4:
        t = torch.flip(t, dims=[-1])
    return torch.rot90(t, k & 3, dims=[-2, -1])


def undihedral_t(t, k):
    t = torch.rot90(t, -(k & 3), dims=[-2, -1])
    if k & 4:
        t = torch.flip(t, dims=[-1])
    return t


@torch.no_grad()
def run_model(model, lrs, device, scale, tta=False, batch=8):
    preds = []
    for i in range(0, len(lrs), batch):
        b = torch.from_numpy(np.stack(lrs[i:i + batch])).unsqueeze(1).to(device)
        if tta:
            acc = None
            for k in range(8):
                p = undihedral_t(model.restore(dihedral_t(b, k)), k).float()
                acc = p if acc is None else acc + p
            out = (acc / 8).clamp_(0.0, 1.0)
        else:
            out = model.restore(b)
        preds.extend(out.float().squeeze(1).cpu().numpy())
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no_lpips", action="store_true",
                    help="skip LPIPS (needs torchvision); PSNR/SSIM only")
    ap.add_argument("--limit", type=int, default=None, help="use fewer val images (dev only)")
    ap.add_argument("--out", default="results/metrics.json")
    a = ap.parse_args()

    gt_dir, lr_dir = Path(a.data) / "GT", Path(a.data) / "NoisyLR"
    _, val_names = split_names(gt_dir)
    if a.limit:
        val_names = val_names[:a.limit]
    gts = [np.load(gt_dir / n) for n in val_names]
    lrs = [np.load(lr_dir / n) for n in val_names]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Validation: {len(val_names)} held-out images, device {device}\n")

    rows = {}
    use_lpips = not a.no_lpips
    rows["bicubic x2 (baseline)"] = summarise(
        [bicubic_upscale(l) for l in lrs], gts, device, with_lpips=use_lpips)

    if a.weights:
        try:
            ck = torch.load(a.weights, map_location=device, weights_only=True)
        except Exception:
            ck = torch.load(a.weights, map_location=device, weights_only=False)
        cfg = ck.get("cfg", {})
        model = build(cfg).to(device).eval()
        model.load_state_dict(ck["model"])
        scale = cfg.get("scale", 2)
        label = f"RestoreNet ({model.n_params():,} params)"
        rows[label] = summarise(run_model(model, lrs, device, scale), gts, device,
                                with_lpips=use_lpips)
        if a.tta:
            rows[label + " + 8x TTA"] = summarise(
                run_model(model, lrs, device, scale, tta=True), gts, device,
                with_lpips=use_lpips)

        # A metric that cannot get worse is not measuring anything: a randomly
        # initialised network of the same shape must score clearly below the trained one.
        rnd = build(cfg).to(device).eval()
        rows["untrained control"] = summarise(
            run_model(rnd, lrs, device, scale), gts, device, with_lpips=False)

    print(f"{'method':<34}{'PSNR (dB)':>11}{'SSIM':>9}{'LPIPS':>9}")
    print("-" * 63)
    for k, v in rows.items():
        cell = f"{v['lpips']:.4f}" if "lpips" in v else "     -"
        print(f"{k:<34}{v['psnr']:>11.3f}{v['ssim']:>9.4f}{cell:>9}")

    if a.weights:
        base, best = rows["bicubic x2 (baseline)"], rows[label]
        if best["psnr"] <= base["psnr"]:
            raise SystemExit(
                f"ABORT: model PSNR {best['psnr']:.3f} does not beat the bicubic "
                f"baseline {base['psnr']:.3f} -- do not ship this checkpoint")
        msg = (f"\nCHECK: model beats bicubic baseline by "
               f"{best['psnr'] - base['psnr']:.3f} dB PSNR, "
               f"{best['ssim'] - base['ssim']:+.4f} SSIM")
        if "lpips" in best and "lpips" in base:
            msg += f", {base['lpips'] - best['lpips']:+.4f} LPIPS (lower is better)"
        print(msg)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"n_val": len(val_names), "results": rows}, indent=2))
    print(f"Written to {a.out}")


if __name__ == "__main__":
    main()
