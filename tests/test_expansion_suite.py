import json
from pathlib import Path

from experiments.experiment_expansion_suite import run


def test_expansion_suite_quick_writes_passing_claims(tmp_path):
    manifest = run(quick=True, output_dir=tmp_path / "expansion")

    assert Path(manifest["summary"]).exists()
    assert Path(manifest["trials"]).exists()
    assert Path(manifest["claims"]).exists()
    assert Path(tmp_path / "expansion" / "manifest.json").exists()
    assert json.loads(Path(manifest["claims"]).read_text(encoding="utf-8"))["all_passed"]
