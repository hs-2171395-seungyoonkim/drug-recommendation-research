"""Stage 1, arm OFF: SafeDrug trained with the DDI loss disabled.

Imports scripts/safedrug_train_dump.py unchanged and overrides one module
constant before calling its main(): TARGET_DDI = 1.0. A visit's DDI rate can
never exceed 1.0, so the rate gate `current_ddi_rate <= target_ddi` is always
true and every step optimises 0.95 * BCE + 0.05 * multilabel-margin only --
the beta blend with loss_ddi is never entered. Nothing else changes: same
original records (HADM_ID order), same seeds, 50 epochs, threshold 0.5,
best-epoch rule, per-visit dump, run_manifest (which records target_ddi=1.0).
The bipartite DDI mask H inside the model architecture is untouched; this arm
removes the *loss*, not the structure.

Arm ON is the existing track-04 baseline (target_ddi 0.06, kp 0.05), whose
threshold-0.5 dumps live in SafeDrug_군집별평가_20260904/개입실험/B_thresholds/seed{k}/global_0.5/.

    py -3.12 scripts/run_safedrug_ddi_off.py --seed 0 [--epochs 50] [--smoke]

--seed 0 = the wrapper's SafeDrug defaults (torch 1203 / numpy 2048 / python
1203), exactly like the track-04 "seed0" run; --seed 1..3 override all three.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "safedrug_train_dump.py"
OUT_ROOT = ROOT / "out" / "safedrug_ddi_off"
TARGET_DDI_OFF = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="0 = wrapper defaults; 1..n = override")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    import torch
    torch.set_num_threads(args.threads)

    spec = importlib.util.spec_from_file_location("safedrug_train_dump", WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["safedrug_train_dump"] = mod
    spec.loader.exec_module(mod)
    assert mod.TARGET_DDI == 0.06, "wrapper default changed; re-check the arm definition"
    mod.TARGET_DDI = TARGET_DDI_OFF   # main() reads the module global at call time

    out_dir = OUT_ROOT / f"seed{args.seed}"
    argv = ["--epochs", str(args.epochs), "--out-dir", str(out_dir)]
    if args.seed != 0:
        argv += ["--seed", str(args.seed)]
    if args.smoke:
        argv.append("--smoke")
    print(f"arm=OFF target_ddi={mod.TARGET_DDI}\nwrapper={WRAPPER}\nout={out_dir}\nargv={argv}", flush=True)
    mod.main(argv)


if __name__ == "__main__":
    main()
