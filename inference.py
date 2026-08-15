"""Restore a directory of degraded .npy images. This is the script KLA benchmarks.

    python inference.py --input_dir <degraded> --output_dir <restored>

Contract:
  in   -- directory of .npy files, float32, single channel, any size, values may lie
          outside [0,1]
  out  -- one .npy per input, SAME filename, float32, exactly `scale`x larger in each
          dimension, values clamped to [0,1]

No source edits required. Model architecture is read from the checkpoint, so there is
no config to keep in sync. Runs on GPU when one is present and falls back to CPU.

Timing is reported the way KLA defines end-to-end runtime: disk read, preprocessing,
host-to-device transfer, model execution, device-to-host transfer, and saving. The
totals below cover all of it, not just the forward pass.
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.model import build


def list_inputs(input_dir):
    """Every .npy in the directory, minus macOS resource-fork junk.

    The provided zips carry __MACOSX/._NAME.npy entries alongside the real files.
    They share basenames with real data and are not valid arrays, so any glob that
    forgets to drop them either crashes or silently doubles the file count.
    """
    files = sorted(p for p in Path(input_dir).glob("*.npy")
                   if not p.name.startswith("._"))
    if not files:
        raise SystemExit(f"ABORT: no .npy files found in {input_dir}")
    return files


def load_model(weights, device):
    try:
        ck = torch.load(weights, map_location=device, weights_only=True)
    except Exception:
        ck = torch.load(weights, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    model = build(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg


def dihedral_t(t, k):
    if k & 4:
        t = torch.flip(t, dims=[-1])
    return torch.rot90(t, k & 3, dims=[-2, -1])


def undihedral_t(t, k):
    t = torch.rot90(t, -(k & 3), dims=[-2, -1])
    if k & 4:
        t = torch.flip(t, dims=[-1])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="weights/model.pt")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--fp32", action="store_true", help="disable half precision")
    ap.add_argument("--tta", action="store_true",
                    help="8x dihedral self-ensemble: slightly better, ~8x slower")
    a = ap.parse_args()

    t_start = time.perf_counter()
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda" and not a.fp32
    torch.backends.cudnn.benchmark = True

    model, cfg = load_model(a.weights, device)
    scale = cfg.get("scale", 2)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_inputs(a.input_dir)

    print(f"Device : {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}"
          f"  half={amp}  tta={a.tta}")
    print(f"Model  : {model.n_params():,} parameters, x{scale}, from {a.weights}")
    print(f"Input  : {len(files)} files from {a.input_dir}")

    # Group by shape so every batch is rectangular. The released test set is uniformly
    # 128x128, but the brief also mentions 256x256 inputs, and a mixed directory must
    # not crash.
    by_shape = defaultdict(list)
    t = time.perf_counter()
    arrays = {}
    for f in files:
        arr = np.load(f).astype(np.float32)
        arrays[f] = arr
        by_shape[arr.shape].append(f)
    t_read = time.perf_counter() - t

    t_transfer = t_model = t_save = 0.0
    written = 0
    for shape, group in by_shape.items():
        for i in range(0, len(group), a.batch_size):
            chunk = group[i:i + a.batch_size]

            t = time.perf_counter()
            batch = torch.from_numpy(np.stack([arrays[f] for f in chunk])).unsqueeze(1)
            batch = batch.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_transfer += time.perf_counter() - t

            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                if a.tta:
                    acc = None
                    for k in range(8):
                        p = undihedral_t(model.restore(dihedral_t(batch, k)), k)
                        acc = p.float() if acc is None else acc + p.float()
                    pred = (acc / 8).clamp_(0.0, 1.0)
                else:
                    pred = model.restore(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_model += time.perf_counter() - t

            t = time.perf_counter()
            out = pred.float().squeeze(1).cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_transfer += time.perf_counter() - t

            t = time.perf_counter()
            for f, arr in zip(chunk, out):
                np.save(out_dir / f.name, arr.astype(np.float32))
                written += 1
            t_save += time.perf_counter() - t

    total = time.perf_counter() - t_start

    # Positive confirmation. Silence is indistinguishable from "the check did not run".
    problems = []
    for f in files:
        o = out_dir / f.name
        if not o.exists():
            problems.append(f"{f.name}: missing"); continue
        arr = np.load(o)
        exp = tuple(s * scale for s in arrays[f].shape)
        if arr.shape != exp:
            problems.append(f"{f.name}: shape {arr.shape}, expected {exp}")
        elif arr.dtype != np.float32:
            problems.append(f"{f.name}: dtype {arr.dtype}, expected float32")
        elif arr.min() < 0.0 or arr.max() > 1.0:
            problems.append(f"{f.name}: range [{arr.min():.3f},{arr.max():.3f}] outside [0,1]")
    if problems:
        for p in problems[:10]:
            print(f"  {p}")
        raise SystemExit(f"ABORT: {len(problems)} output(s) violate the contract")

    print(f"\nCHECK: {written} outputs written to {out_dir}, all x{scale}, "
          f"float32, within [0,1], filenames preserved")
    print("\nEnd-to-end timing (all stages KLA counts):")
    print(f"  disk read + preprocess : {t_read:7.3f} s")
    print(f"  host<->device transfer : {t_transfer:7.3f} s")
    print(f"  model execution        : {t_model:7.3f} s")
    print(f"  save to disk           : {t_save:7.3f} s")
    print(f"  ---------------------------------")
    print(f"  TOTAL end-to-end       : {total:7.3f} s   "
          f"({total / len(files) * 1000:.2f} ms/image, {len(files) / total:.1f} images/s)")
    sync_note = (", with torch.cuda.synchronize() around every GPU stage"
                 if device.type == "cuda" else " (CPU run, no GPU sync needed)")
    print(f"  batch size {a.batch_size}, timed with time.perf_counter(){sync_note}")


if __name__ == "__main__":
    main()
