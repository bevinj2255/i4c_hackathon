"""Rebuild the complete submission from a checkpoint, then check it against KLA's list.

    python package_submission.py --checkpoint weights/base_best.pt

Runs, in order: promote checkpoint -> metrics -> restored test outputs -> figures ->
README tables -> deck. Then verifies every mandatory deliverable exists and ABORTS if
one is missing.

This exists because the submission has eight required pieces and rebuilding them by
hand every time a better checkpoint appears is how one quietly goes stale. Running it
end to end is cheap; discovering at the deadline that results/ still holds last run's
numbers is not.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd, label):
    print(f"\n=== {label} ===", flush=True)
    r = subprocess.run([sys.executable] + cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"ABORT: {label} failed (exit {r.returncode})")


def fill_readme(metrics, hardware):
    """Rewrite the two placeholder tables in README.md from measured numbers."""
    p = ROOT / "README.md"
    text = p.read_text(encoding="utf-8")

    def row(match_key, label):
        for k, v in metrics.items():
            if match_key.lower() in k.lower():
                lp = f"{v['lpips']:.4f}" if "lpips" in v else "—"
                return f"| {label} | {v['psnr']:.3f} | {v['ssim']:.4f} | {lp} |"
        return f"| {label} | _pending_ | _pending_ | _pending_ |"

    results_table = "\n".join([
        "| Method | PSNR (dB) | SSIM | LPIPS |",
        "|---|---|---|---|",
        row("bicubic", "Bicubic ×2 (baseline)"),
        row("RestoreNet (", "RestoreNet (ours)"),
        row("TTA", "RestoreNet + 8× TTA"),
        row("untrained", "Untrained control"),
    ])
    text = re.sub(r"\| Method \| PSNR \(dB\) \| SSIM \| LPIPS \|\n\|---\|---\|---\|---\|\n(\|.*\n)+",
                  results_table + "\n", text)

    hw_table = "\n".join(
        ["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in hardware.items()])
    text = re.sub(r"\| \| \|\n\|---\|---\|\n\| Training hardware \|(.*\n)+?\n",
                  hw_table + "\n\n", text)
    p.write_text(text, encoding="utf-8")
    print("  README tables updated from measured values")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--team", default="TEAM NAME")
    ap.add_argument("--members", default="MEMBER 1 (role), MEMBER 2 (role)")
    ap.add_argument("--college", default="COLLEGE NAME")
    ap.add_argument("--contact", default="EMAIL / PHONE")
    ap.add_argument("--skip_tta", action="store_true",
                    help="skip the 8x TTA evaluation (it is slow)")
    a = ap.parse_args()

    ck = Path(a.checkpoint)
    if not ck.exists():
        raise SystemExit(f"ABORT: no checkpoint at {ck}")

    # 1. Promote to the stable name inference.py defaults to.
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy2(ck, ROOT / "weights" / "model.pt")
    print(f"CHECK: {ck.name} -> weights/model.pt")

    # 2. Metrics vs baseline. evaluate.py aborts if the model loses to bicubic.
    ev = ["evaluate.py", "--weights", "weights/model.pt"]
    if not a.skip_tta:
        ev.append("--tta")
    run(ev, "metrics on held-out validation split")

    # 3. Restored outputs for the released test set -- a required deliverable.
    run(["inference.py", "--input_dir", "data/test/NoisyLR",
         "--output_dir", "results/restored_test"], "restoring the 397 test images")

    # 4. Figures, including the required failure case.
    run(["make_figures.py", "--weights", "weights/model.pt"], "figures")

    # 5. README tables.
    metrics = json.loads((ROOT / "results/metrics.json").read_text())["results"]
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logs = [p for p in (ROOT / "results").glob("*_log.csv") if p.stem != "smoke_log"]
    epochs = hours = 0
    if logs:
        import csv as _csv
        rows = list(_csv.DictReader(max(logs, key=lambda p: p.stat().st_mtime).open()))
        epochs = len(rows)
        hours = sum(float(r["seconds"]) for r in rows) / 3600.0
    fill_readme(metrics, {
        "Training hardware": gpu,
        "Training time": f"{epochs} epochs, {hours:.1f} h wall clock",
        "Inference hardware measured": gpu,
        "End-to-end runtime": "see the timing block printed by `inference.py` "
                              "(disk read, preprocessing, host↔device transfer, model "
                              "execution, saving)",
        "Batch size": "16",
        "Timing method": "`time.perf_counter()` with `torch.cuda.synchronize()` around "
                         "every GPU stage",
        "Seed": "0 (set for `random`, `numpy` and `torch`; recorded in the checkpoint)",
    })

    # 6. Deck.
    run(["make_ppt.py", "--team", a.team, "--members", a.members,
         "--college", a.college, "--contact", a.contact, "--gpu", gpu], "submission deck")

    # 7. The checklist, as an assertion rather than a hope.
    print("\n=== KLA mandatory deliverables ===")
    required = {
        "README.md": ROOT / "README.md",
        "requirements.txt": ROOT / "requirements.txt",
        "inference script (input_dir/output_dir)": ROOT / "inference.py",
        "training script": ROOT / "train.py",
        "model weights": ROOT / "weights/model.pt",
        "restored test outputs": ROOT / "results/restored_test",
        "metrics (PSNR/SSIM/LPIPS)": ROOT / "results/metrics.json",
        "figures incl. failure case": ROOT / "results/figures/restored_worst_1.png",
        "solution deck": next(iter((ROOT / "results").glob("*_KLA_PS01.pptx")), Path("missing")),
    }
    missing = []
    for label, path in required.items():
        n = len(list(path.glob("*.npy"))) if path.is_dir() else None
        if not path.exists():
            missing.append(label)
            print(f"  MISSING  {label}")
        else:
            extra = f" ({n} files)" if n is not None else ""
            print(f"  ok       {label}{extra}")
    n_out = len(list((ROOT / "results/restored_test").glob("*.npy")))
    n_in = len(list((ROOT / "data/test/NoisyLR").glob("*.npy")))
    if n_out != n_in:
        missing.append(f"restored outputs {n_out} != {n_in} inputs")
    if missing:
        raise SystemExit(f"\nABORT: incomplete submission -- {missing}")
    print(f"\nPASS: submission package complete ({n_out}/{n_in} test images restored).")


if __name__ == "__main__":
    main()
