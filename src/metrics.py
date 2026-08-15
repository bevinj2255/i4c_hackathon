"""PSNR, SSIM and LPIPS -- the three KLA combines into the quality score.

All three are computed on the saved-image contract: float32, single channel, values
in [0,1]. data_range is pinned to 1.0 everywhere rather than inferred from the array,
because inferring it from each image's own min/max would quietly change the units per
image and make the numbers incomparable.
"""
import numpy as np

_LPIPS = None


def psnr(pred, gt, data_range=1.0):
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(pred, gt, data_range=1.0):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(gt, pred, data_range=data_range))


def lpips_batch(pred, gt, device="cpu"):
    """LPIPS over a batch of (N,H,W) arrays. Lower is better.

    LPIPS expects 3-channel images in [-1,1]; grayscale is replicated across the three
    channels, which is the standard way single-channel data is fed to it.
    """
    global _LPIPS
    import torch
    import lpips as lpips_lib
    if _LPIPS is None:
        _LPIPS = lpips_lib.LPIPS(net="alex", verbose=False).to(device).eval()

    def prep(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).float().unsqueeze(1)
        return (t.repeat(1, 3, 1, 1) * 2.0 - 1.0).to(device)

    with torch.no_grad():
        d = _LPIPS(prep(pred), prep(gt))
    return d.flatten().cpu().numpy().astype(np.float64)


def summarise(preds, gts, device="cpu", with_lpips=True):
    """preds/gts: lists of 2-D float arrays in [0,1]. Returns mean metrics."""
    out = {
        "psnr": float(np.mean([psnr(p, g) for p, g in zip(preds, gts)])),
        "ssim": float(np.mean([ssim(p, g) for p, g in zip(preds, gts)])),
        "n": len(preds),
    }
    if with_lpips:
        vals = []
        for i in range(0, len(preds), 16):
            vals.append(lpips_batch(np.stack(preds[i:i + 16]),
                                    np.stack(gts[i:i + 16]), device))
        out["lpips"] = float(np.mean(np.concatenate(vals)))
    return out


def _demo():
    """A metric that cannot get worse is not measuring anything."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64] / 63.0
    gt = ((np.sin(xx * 9) * np.cos(yy * 7) + 1) / 2).astype(np.float32)

    assert psnr(gt, gt) == float("inf"), "identical images must give infinite PSNR"
    assert abs(ssim(gt, gt) - 1.0) < 1e-6, "identical images must give SSIM 1"

    # Degrading the prediction must move both metrics the right way -- this is the
    # mutation: if these stay put, the metric is a green light wired to nothing.
    mild = np.clip(gt + rng.normal(0, 0.02, gt.shape), 0, 1).astype(np.float32)
    harsh = np.clip(gt + rng.normal(0, 0.20, gt.shape), 0, 1).astype(np.float32)
    assert psnr(mild, gt) > psnr(harsh, gt), "PSNR did not fall when noise rose"
    assert ssim(mild, gt) > ssim(harsh, gt), "SSIM did not fall when noise rose"

    print(f"CHECK: metrics.py self-check passed "
          f"(mild {psnr(mild, gt):.2f}dB/{ssim(mild, gt):.4f} vs "
          f"harsh {psnr(harsh, gt):.2f}dB/{ssim(harsh, gt):.4f})")


if __name__ == "__main__":
    _demo()
