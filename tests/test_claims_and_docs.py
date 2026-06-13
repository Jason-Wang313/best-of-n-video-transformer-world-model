from __future__ import annotations

from pathlib import Path

from counterfactual_video_audit.claims import find_forbidden_claims
from counterfactual_video_audit.config import FINAL_AUDIT_STATUSES


def test_claim_audit_catches_forbidden_claim(tmp_path: Path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "paper").mkdir()
    (tmp_path / "README.md").write_text("This solves robot planning in every setting.", encoding="utf-8")
    hits = find_forbidden_claims(tmp_path)
    assert hits
    assert hits[0]["phrase"] == "solves robot planning"


def test_final_audit_status_is_exactly_one_allowed_label():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "final_audit.md").read_text(encoding="utf-8")
    matches = [status for status in FINAL_AUDIT_STATUSES if status in text]
    assert matches == ["submission-ready v2"]


def test_required_docs_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "docs/claims.md",
        "docs/differentiation_from_wam.md",
        "docs/differentiation_from_prior_projects.md",
        "docs/reviewer_attacks.md",
        "docs/final_audit.md",
        "paper/abstract.md",
        "paper/intro.md",
        "paper/method.md",
        "paper/theory.md",
        "paper/experiments.md",
        "paper/related_work.md",
        "paper/limitations.md",
        "paper/checklist.md",
    ]:
        assert (root / rel).exists(), rel
