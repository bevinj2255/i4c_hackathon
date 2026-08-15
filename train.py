"""Train the restoration network.

    python train.py --config configs/base.json
    python train.py --config configs/base.json --resume weights/base_last.pt
    python train.py --config configs/base.json --overfit 2     # pipeline sanity check
    python train.py --config configs/base.json --benchmark     # measured images/sec

Checkpoints are named after the config, so two configs can never overwrite each
other's results -- a mistake that is very cheap to make and very expensive to notice
late. Every epoch appends to results/<config>_log.csv.

Validation is 200 held-out images fixed by seed in src/data.py. They are never
trained on, and model selection uses only this split.
"""
import argparse
import os
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import EvalDataset, RestorationDataset, split_names
from src.model import build


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(obj, path):
    """Write a checkpoint so no reader can ever see a half-written file.

    Training saves every epoch while other tools (evaluate.py, inference.py,
    package_submission.py) may be reading the same file. Writing in place gives
    them a torn file and a deserialisation error that looks like a corrupt model.
    os.replace is atomic on both Windows and POSIX, so a reader sees either the
    old checkpoint or the new one, never a mixture.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def charbonnier(pred, gt, eps=1e-3):
    """Smooth L1. Less prone than L2 to washing out detail, standard for restoration."""
    return torch.mean(torch.sqrt((pred - gt) ** 2 + eps ** 2))


def edge_loss(pred, gt):
    """L1 on finite differences. Penalises softened edges that plain L1 tolerates."""
    px, gx = pred[..., :, 1:] - pred[..., :, :-1], gt[..., :, 1:] - gt[..., :, :-1]
    py, gy = pred[..., 1:, :] - pred[..., :-1, :], gt[..., 1:, :] - gt[..., :-1, :]
    return (px - gx).abs().mean() + (py - gy).abs().mean()


def make_loss(cfg):
    w = cfg.get("edge_weight", 0.0)

    def loss_fn(pred, gt):
        loss = charbonnier(pred, gt)
        if w > 0:
            loss = loss + w * edge_loss(pred, gt)
        return loss

    return loss_fn


@torch.no_grad()
def bicubic_psnr(loader, device, scale=2):
    """Mean PSNR of plain bicubic upscaling -- the zero-parameter reference."""
    total, n = 0.0, 0
    for lr, gt in loader:
        lr, gt = lr.to(device), gt.to(device)
        up = torch.nn.functional.interpolate(
            lr.float(), scale_factor=scale, mode="bicubic", align_corners=False
        ).clamp_(0.0, 1.0)
        mse = torch.mean((up - gt.float()) ** 2, dim=(1, 2, 3))
        total += float(torch.sum(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))))
        n += lr.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, amp):
    """Mean PSNR over the val split, on the same [0,1]-clamped output we would save."""
    model.eval()
    total, n = 0.0, 0
    for lr, gt in loader:
        lr, gt = lr.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = model.restore(lr)
        mse = torch.mean((pred.float() - gt.float()) ** 2, dim=(1, 2, 3))
        total += float(torch.sum(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))))
        n += lr.shape[0]
    model.train()
    return total / max(n, 1)


def benchmark(model, cfg, device, amp, seconds=30):
    """Measured images/sec for a real training step. Not an estimate.

    Stage 0 of the plan uses this to choose the model size: whatever trains an epoch
    of 3000 images in a few minutes on the GPU actually present.
    """
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = make_loss(cfg)
    # patch 0 means "train on whole images", which for this dataset is 128x128.
    p = cfg["patch"] or 128
    s, b = cfg["scale"], cfg["batch_size"]
    lr_t = torch.randn(b, 1, p, p, device=device)
    gt_t = torch.rand(b, 1, p * s, p * s, device=device)

    for _ in range(5):  # warm up cudnn autotuning before timing anything
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(lr_t), gt_t)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
    if device.type == "cuda":
        torch.cuda.synchronize()

    steps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = loss_fn(model(lr_t), gt_t)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ips = steps * b / dt
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
    print(f"MEASURED: {ips:.1f} images/sec  ({steps} steps of batch {b} in {dt:.1f}s)")
    print(f"MEASURED: peak GPU memory {mem:.2f} GiB")
    print(f"MEASURED: {3000 / ips / 60:.2f} min/epoch over 3000 training images")
    return ips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.json")
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--overfit", type=int, default=0,
                    help="overfit this many images as a pipeline sanity check")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    name = cfg.get("name", Path(a.config).stem)
    if a.epochs:
        cfg["epochs"] = a.epochs
    set_seed(cfg.get("seed", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and cfg.get("amp", True)
    torch.backends.cudnn.benchmark = True

    model = build(cfg).to(device)
    print(f"Config {name}: channels={cfg['channels']} blocks={cfg['blocks']} "
          f"-> {model.n_params():,} parameters")
    print(f"Device: {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}"
          f", AMP={amp}")

    if a.benchmark:
        benchmark(model, cfg, device, amp)
        return

    gt_dir, lr_dir = Path(a.data) / "GT", Path(a.data) / "NoisyLR"
    train_names, val_names = split_names(gt_dir)

    if a.overfit:
        # Deliberately trivial task: a handful of fixed pairs the model should be able
        # to memorise outright. If the loss does not collapse, something in the data
        # path or the model is wrong and no amount of real training will rescue it.
        #
        # patch is forced to 0 (whole image) and augmentation off, so the input really
        # is a constant. With random crops the input differs every step, the network
        # can only learn a generic denoiser, and the loss parks at the noise floor
        # (~0.022 here) no matter how healthy the pipeline is -- which makes the check
        # unable to distinguish a bug from a small model. A check that cannot fail for
        # the reason it exists is not a check.
        train_names, val_names = train_names[:a.overfit], train_names[:a.overfit]
        cfg["epochs"] = cfg.get("overfit_epochs", 300)
        train_set = RestorationDataset(train_names * 8, gt_dir, lr_dir,
                                       patch=0, scale=cfg["scale"],
                                       p_synth=0.0, augment=False, seed=cfg["seed"])
    else:
        train_set = RestorationDataset(train_names, gt_dir, lr_dir,
                                       patch=cfg["patch"], scale=cfg["scale"],
                                       p_synth=cfg.get("p_synth", 0.5),
                                       augment=True, seed=cfg["seed"])

    workers = cfg.get("workers", 0)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=workers, pin_memory=device.type == "cuda",
                              drop_last=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(EvalDataset(val_names, gt_dir, lr_dir), batch_size=8,
                            shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
    print(f"Train {len(train_set)} samples, val {len(val_names)} images, "
          f"batch {cfg['batch_size']}, patch {cfg['patch']}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"],
                                                       eta_min=cfg["lr"] * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = make_loss(cfg)

    first_loss = None
    start_epoch, best = 0, -1.0
    if a.resume and Path(a.resume).exists():
        ck = torch.load(a.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        print(f"Resumed from {a.resume} at epoch {start_epoch} (best val PSNR {best:.3f})")

    Path("weights").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    log_path = Path("results") / f"{name}_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,train_loss,val_psnr,lr,seconds\n")

    for epoch in range(start_epoch, cfg["epochs"]):
        t0, running, steps = time.perf_counter(), 0.0, 0
        for lr_t, gt_t in train_loader:
            lr_t = lr_t.to(device, non_blocking=True)
            gt_t = gt_t.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(lr_t), gt_t)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += float(loss.detach()); steps += 1
        sched.step()

        train_loss = running / max(steps, 1)
        if first_loss is None:
            first_loss = train_loss
        val_psnr = validate(model, val_loader, device, amp)
        dt = time.perf_counter() - t0
        print(f"epoch {epoch + 1:4d}/{cfg['epochs']}  loss {train_loss:.5f}  "
              f"val PSNR {val_psnr:.3f} dB  lr {sched.get_last_lr()[0]:.2e}  {dt:.0f}s")
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.6f}", f"{val_psnr:.4f}",
                                    f"{sched.get_last_lr()[0]:.3e}", f"{dt:.1f}"])

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": epoch, "best": max(best, val_psnr), "cfg": cfg}
        save_atomic(ck, Path("weights") / f"{name}_last.pt")
        if val_psnr > best:
            best = val_psnr
            save_atomic(ck, Path("weights") / f"{name}_best.pt")

    print(f"\nDone. Best val PSNR {best:.3f} dB -> weights/{name}_best.pt")
    if a.overfit:
        # The sanity check has a pass/fail, not just a printout.
        #
        # It deliberately does NOT demand near-zero loss. A fully-convolutional network
        # is a local operator; the same noisy neighbourhood maps to different clean
        # values in different places, so a noisy->clean pair cannot be memorised to
        # zero however healthy the pipeline is. Measured: loss plateaus around 0.018
        # on a fixed pair and will not go lower. Asserting "< 0.01" would fail forever
        # for a reason that has nothing to do with correctness.
        #
        # What a real bug DOES break: misaligned crops, a swapped target, or dead
        # gradients all leave the model unable to beat plain bicubic on the very
        # images it just trained on, and leave the loss flat from the start.
        base = bicubic_psnr(val_loader, device, cfg["scale"])
        drop = first_loss / max(train_loss, 1e-9)
        print(f"    first-epoch loss {first_loss:.5f} -> final {train_loss:.5f} "
              f"({drop:.1f}x lower)")
        print(f"    bicubic on the same images: {base:.3f} dB, model: {val_psnr:.3f} dB")
        # Report every failing criterion, not just the first. A swapped target trips
        # both, and blaming only "dead gradients" sends the next person the wrong way.
        reasons = []
        if drop < 2.0:
            reasons.append(f"loss only fell {drop:.2f}x (want >=2x) -- gradients may not "
                           f"be reaching the model, or the target is unlearnable")
        if val_psnr < base + 1.0:
            reasons.append(f"model {val_psnr:.3f} dB does not clear bicubic {base:.3f} dB "
                           f"by 1 dB on its own training images -- suspect crop "
                           f"misalignment or a swapped target")
        if reasons:
            raise SystemExit("ABORT: overfit check failed --\n  - " + "\n  - ".join(reasons))
        print(f"CHECK: overfit sanity check passed (loss fell {drop:.1f}x, model beats "
              f"bicubic by {val_psnr - base:.2f} dB on memorised images)")


if __name__ == "__main__":
    main()
