"""Run the experiment suite in one Python process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.exact_law_validation import run as run_exact
from experiments.experiment_learned_video_transformer import run as run_learned
from experiments.experiment_occlusion_stress import run as run_occlusion
from experiments.experiment_video_tail_failure import run as run_failure
from experiments.experiment_video_diagnostics import run as run_diagnostics
from experiments.experiment_video_repairs import run as run_repairs
from experiments.multiseed_evidence import run as run_multiseed
from scripts.run_claim_audit import main as run_claim_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed-base", type=int, default=20)
    args = parser.parse_args()
    steps = [
        ("experiments/exact_law_validation.py", lambda: run_exact(smoke=args.smoke, seed=args.seed_base + 1)),
        (
            "experiments/experiment_learned_video_transformer.py",
            lambda: run_learned(smoke=args.smoke, seed=args.seed_base + 2),
        ),
        (
            "experiments/experiment_video_tail_failure.py",
            lambda: run_failure(smoke=args.smoke, seed=args.seed_base + 3),
        ),
        ("experiments/experiment_video_repairs.py", lambda: run_repairs(smoke=args.smoke, seed=args.seed_base + 4)),
        (
            "experiments/experiment_video_diagnostics.py",
            lambda: run_diagnostics(smoke=args.smoke, seed=args.seed_base + 5),
        ),
        (
            "experiments/experiment_occlusion_stress.py",
            lambda: run_occlusion(smoke=args.smoke, seed=args.seed_base + 6),
        ),
        ("experiments/multiseed_evidence.py", lambda: run_multiseed(smoke=args.smoke, seed=args.seed_base + 100)),
    ]
    for label, fn in steps:
        print(f"[suite] running {label}", flush=True)
        fn()
    print("[suite] running scripts/run_claim_audit.py", flush=True)
    run_claim_audit()


if __name__ == "__main__":
    main()
