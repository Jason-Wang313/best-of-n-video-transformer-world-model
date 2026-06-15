from __future__ import annotations

import json
from pathlib import Path

from experiments.experiment_moving_mnist_benchmark import run
from counterfactual_video_audit.moving_mnist_benchmark import load_digit_bank


def test_moving_mnist_fallback_digits_are_available(tmp_path: Path):
    digits, source = load_digit_bank(tmp_path / "cache", allow_download=False)

    assert source == "template_fallback"
    assert digits.shape[1:] == (28, 28)
    assert float(digits.max()) > 0.0


def test_moving_mnist_quick_claims_pass_without_network(tmp_path: Path):
    manifest = run(quick=True, output_dir=tmp_path / "moving_mnist", allow_download=False)
    claims = json.loads(Path(manifest["claims"]).read_text(encoding="utf-8"))

    assert Path(manifest["metrics"]).exists()
    assert Path(manifest["aggregate_metrics"]).exists()
    assert claims["all_passed"]
