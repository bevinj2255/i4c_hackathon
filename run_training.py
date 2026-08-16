"""One command that sets up and runs training on a new machine.

    python run_training.py

Does everything that is easy to get wrong, automatically:

  1. checks PyTorch can see the GPU, and prints the exact fix if not
  2. unpacks the KLA zips if the data is not there yet
  3. verifies the degradation model still matches the real data (aborts if not)
  4. picks the model size from your VRAM
  5. BENCHMARKS fp16 against fp32 and keeps the faster one -- this is not a
     preference, it was measured 5x either way on different cards
  6. runs the overfit sanity check so a broken pipeline is caught in 2 minutes
     rather than 4 hours
  7. starts training, resuming automatically if a previous run was interrupted

Options:
    --dry_run     do everything except start the actual training
    --config X    force a specific config instead of choosing by VRAM
    --hours H     size the run to roughly H hours (default 4)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def say(msg):
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}", flush=True)


def die(msg):
    print(f"\nSTOPPED: {msg}\n", flush=True)
    sys.exit(1)


def run(args, label):
    print(f"\n--- {label} ---", flush=True)
    r = subprocess.run([sys.executable] + args, cwd=ROOT)
    if r.returncode != 0:
        die(f"{label} failed. Fix the error above before continuing.")


def check_gpu():
    say("STEP 1 of 6  -  checking your GPU")
    try:
        import torch
    except ImportError:
        die("PyTorch is not installed. Run:\n"
            "    pip install -r requirements.txt\n"
            "  and on Windows also:\n"
            "    pip install torch==2.9.1 torchvision "
            "--index-url https://download.pytorch.org/whl/cu128")
    if not torch.cuda.is_available():
        die("PyTorch cannot see your GPU (it is probably the CPU-only build).\n"
            "  Install the CUDA build:\n"
            "    pip install torch==2.9.1 torchvision "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            "  Then check it prints True:\n"
            '    python -c "import torch; print(torch.cuda.is_available())"')
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    print(f"  GPU   : {name}")
    print(f"  VRAM  : {vram:.1f} GiB")
    print(f"  torch : {torch.__version__}")
    return name, vram


def check_data():
    say("STEP 2 of 6  -  checking the dataset")
    gt = ROOT / "data/train/GT"
    if gt.exists() and len(list(gt.glob("*.npy"))) >= 3000:
        print(f"  data already prepared ({len(list(gt.glob('*.npy')))} training pairs)")
        return
    zips = [ROOT / "train.zip", ROOT / "Test_NoisyLR.zip"]
    missing = [z.name for z in zips if not z.exists()]
    if missing:
        die(f"Missing {missing}.\n"
            f"  Download train.zip and Test_NoisyLR.zip from the KLA dataset link and\n"
            f"  put them in this folder ({ROOT}). Do not unzip them yourself.\n"
            f"  You need about 2 GB of free space.")
    print("  unpacking (takes a couple of minutes)...")
    run(["prepare_data.py"], "unpacking the dataset")


def verify():
    say("STEP 3 of 6  -  verifying the data and the degradation model")
    run(["verify_degradation.py"], "degradation check")


def choose_config(vram, forced):
    say("STEP 4 of 6  -  choosing the model size")
    if forced:
        cfg = ROOT / forced
        print(f"  using {cfg.name} (forced)")
        return cfg
    if vram >= 7.0:
        cfg = ROOT / "configs/large_perceptual.json"
        print(f"  {vram:.1f} GiB VRAM -> large_perceptual (96 channels, 24 blocks, 4.4M params)")
    elif vram >= 5.0:
        cfg = ROOT / "configs/large.json"
        print(f"  {vram:.1f} GiB VRAM -> large (96 channels, 24 blocks)")
    else:
        cfg = ROOT / "configs/perceptual.json"
        print(f"  {vram:.1f} GiB VRAM -> perceptual (64 channels, 16 blocks)")
    return cfg


def benchmark(cfg_path, hours):
    say("STEP 5 of 6  -  measuring which precision is faster on YOUR card")
    print("  fp16 is faster on cards with tensor cores and much SLOWER on cards without.")
    print("  Measuring both rather than guessing. Takes about a minute.\n")

    results = {}
    for amp in (True, False):
        cfg = json.loads(cfg_path.read_text())
        cfg["amp"] = amp
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        out = subprocess.run([sys.executable, "train.py", "--config", str(cfg_path),
                              "--benchmark"], cwd=ROOT, capture_output=True, text=True)
        ips = mins = None
        for line in out.stdout.splitlines():
            if "images/sec" in line:
                ips = float(line.split()[1])
            if "min/epoch" in line:
                mins = float(line.split()[1])
        if ips is None:
            print(f"  {'fp16' if amp else 'fp32'}: failed (likely out of memory)")
            if out.stderr:
                print("   ", out.stderr.strip().splitlines()[-1][:100])
            continue
        results[amp] = (ips, mins)
        print(f"  {'fp16' if amp else 'fp32'}: {ips:6.1f} images/sec, {mins:5.2f} min/epoch")

    if not results:
        die("Both precisions failed. Your GPU may not have enough memory.\n"
            "  Open the config and lower \"batch_size\" to 8 or 4, then rerun.")

    best_amp = max(results, key=lambda k: results[k][0])
    ips, mins = results[best_amp]
    epochs = max(20, int(hours * 60 / mins))
    cfg = json.loads(cfg_path.read_text())
    cfg["amp"] = best_amp
    cfg["epochs"] = epochs
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"\n  KEEPING {'fp16' if best_amp else 'fp32'} ({ips:.1f} images/sec)")
    print(f"  Set epochs to {epochs} so the run takes about {epochs * mins / 60:.1f} hours.")
    print(f"  The schedule anneals over exactly that many epochs, so let it finish.")
    return cfg["name"], epochs, mins


def sanity(cfg_path):
    say("STEP 6 of 6  -  2-minute sanity check before committing hours")
    run(["train.py", "--config", str(cfg_path), "--overfit", "2"], "overfit check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--skip_benchmark", action="store_true",
                    help="reuse the precision already in the config "
                         "(for re-runs, or when the GPU is busy)")
    a = ap.parse_args()

    print("\nKLA PS01 - automated training setup")
    print("Nothing here needs decisions from you; it measures and chooses.")

    name, vram = check_gpu()
    check_data()
    verify()
    cfg_path = choose_config(vram, a.config)
    if a.skip_benchmark:
        cfg = json.loads(cfg_path.read_text())
        run_name, epochs, mins = cfg["name"], cfg["epochs"], 0.0
        print(f"\n  (--skip_benchmark: keeping amp={cfg['amp']}, epochs={epochs})")
    else:
        run_name, epochs, mins = benchmark(cfg_path, a.hours)
    if not a.skip_benchmark:
        sanity(cfg_path)

    last = ROOT / "weights" / f"{run_name}_last.pt"
    cmd = ["train.py", "--config", str(cfg_path)]
    if last.exists():
        cmd += ["--resume", str(last)]
        print(f"\n  Found an interrupted run -- resuming from {last.name}")

    say("READY")
    print(f"  GPU        : {name} ({vram:.1f} GiB)")
    print(f"  config     : {cfg_path.name}")
    print(f"  epochs     : {epochs}" + (f"  (~{epochs * mins / 60:.1f} hours)" if mins else ""))
    print(f"\n  When it finishes, send back these two files:")
    print(f"      weights/{run_name}_best.pt")
    print(f"      results/{run_name}_log.csv")
    print(f"\n  git add weights/{run_name}_best.pt results/{run_name}_log.csv")
    print(f"  git commit -m \"{run_name} run on {name}\"")
    print(f"  git push -u origin friend-run")
    print(f"\n  If it stops early, just run this same script again -- it resumes.")

    if a.dry_run:
        print("\n  (--dry_run: not starting training)")
        return
    print("\nStarting training now. You can leave it; it saves after every epoch.\n")
    time.sleep(3)
    run(cmd, "training")


if __name__ == "__main__":
    main()
