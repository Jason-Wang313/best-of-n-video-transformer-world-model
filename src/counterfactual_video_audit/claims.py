"""Claim audit for the scoped counterfactual video audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from counterfactual_video_audit.config import FINAL_AUDIT_STATUSES, GATE_LABELS

FORBIDDEN_CLAIMS = [
    "solves video world modeling",
    "solves robot planning",
    "top-score selection always hurts",
    "video transformers are bad",
    "action consistency always fixes it",
    "real-robot validation",
]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _status(ok: bool, partial: bool = False) -> str:
    if ok:
        return "SUPPORTED"
    return "PARTIAL" if partial else "UNSUPPORTED"


def _text_corpus(root: Path) -> dict[str, str]:
    paths = [root / "README.md"]
    paths.extend(sorted((root / "docs").glob("*.md")))
    paths.extend(sorted((root / "paper").glob("*.md")))
    paths.extend(sorted((root / "paper").glob("*.tex")))
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8") for path in paths if path.exists()}


def find_forbidden_claims(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel, text in _text_corpus(root).items():
        lowered = text.lower()
        for phrase in FORBIDDEN_CLAIMS:
            if phrase.lower() in lowered:
                hits.append({"path": rel, "phrase": phrase})
    return hits


def _final_audit_status(root: Path) -> str | None:
    path = root / "docs" / "final_audit.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    for status in FINAL_AUDIT_STATUSES:
        if status in text:
            return status
    return None


def build_claim_status(root: Path) -> dict[str, Any]:
    results = root / "results"
    exact = _load(results / "exact_law_validation.json")
    failure = _load(results / "experiment_video_tail_failure.json")
    repairs = _load(results / "experiment_video_repairs.json")
    diagnostics = _load(results / "experiment_video_diagnostics.json")
    learned = _load(results / "experiment_learned_video_transformer.json")
    occlusion = _load(results / "experiment_occlusion_stress.json")
    multi = _load(results / "multiseed_strong_evidence.json")

    full_multiseed = bool(multi and not multi.get("smoke"))
    claims: list[dict[str, Any]] = []
    weak_reasons: list[str] = []

    exact_ok = bool(exact and exact.get("max_abs_error", 1.0) < 0.06)
    claims.append(
        {
            "category": "finite law",
            "claim": "The tie-aware finite-pool top-score expectation matches brute-force enumeration and Monte Carlo validation within sampling error.",
            "status": _status(exact_ok),
            "evidence_strength": "STRONG" if exact_ok else "WEAK",
            "evidence": "results/exact_law_validation.json; figures/figure5_exact_law_validation.png",
        }
    )

    key = failure.get("key_result", {}) if failure else {}
    failure_single = bool(
        key
        and key.get("raw_plausibility_delta_high_n", 0.0) > 0.05
        and key.get("raw_action_violation_delta_high_n", 0.0) > 0.15
        and key.get("raw_real_utility_delta_high_n", 1.0) < 0.05
    )
    failure_strong = bool(
        multi
        and multi.get("failure", {}).get("raw_plausibility_delta_high_n_lo", 0.0) > 0.04
        and multi.get("failure", {}).get("raw_action_violation_delta_high_n_lo", 0.0) > 0.12
        and multi.get("failure", {}).get("raw_real_utility_delta_high_n_hi", 1.0) < 0.05
    )
    if full_multiseed and not failure_strong:
        weak_reasons.append("multi-seed video failure margins are below the strong thresholds")
    claims.append(
        {
            "category": "video tail failure",
            "claim": "In the controlled video world, visually top-ranked futures become more plausible as N grows while executed action-conditioned utility stagnates or worsens.",
            "status": _status(failure_strong if full_multiseed else failure_single, partial=bool(failure)),
            "evidence_strength": "STRONG" if failure_strong else ("SMOKE" if failure_single else "WEAK"),
            "evidence": "results/experiment_video_tail_failure.json; figures/figure1_counterfactual_video_lineup.png",
        }
    )

    learned_ok = bool(
        learned
        and (root / "results" / "learned_tiny_video_transformer.pt").exists()
        and learned.get("final_loss", 1e9) < learned.get("initial_loss", 1e9)
    )
    claims.append(
        {
            "category": "learned video scorer",
            "claim": "A compact CPU-trained transformer over video patches is trained and saved as a tiny learned scorer artifact for smoke-scale checks.",
            "status": _status(learned_ok, partial=bool(learned)),
            "evidence_strength": "STRONG" if learned_ok else "WEAK",
            "evidence": "results/experiment_learned_video_transformer.json; results/learned_tiny_video_transformer.pt",
        }
    )

    repair_key = repairs.get("key_result", {}) if repairs else {}
    repair_single = bool(repair_key and repair_key.get("calibrated_repair_n64_improvement_over_raw", 0.0) > 0.35)
    repair_strong = bool(
        multi and multi.get("repair", {}).get("calibrated_repair_n64_improvement_over_raw_lo", 0.0) > 0.25
    )
    if full_multiseed and not repair_strong:
        weak_reasons.append("repair ladder improvement is not strong across seeds")
    claims.append(
        {
            "category": "repair ladder",
            "claim": "Video-specific checks and a small eval-disjoint pilot calibration recover utility over raw visual selection in this controlled setting.",
            "status": _status(repair_strong if full_multiseed else repair_single, partial=bool(repairs)),
            "evidence_strength": "STRONG" if repair_strong else ("SMOKE" if repair_single else "WEAK"),
            "evidence": "results/experiment_video_repairs.json; figures/figure2_repair_ladder.png",
        }
    )

    gate_labels = diagnostics.get("gate_labels", []) if diagnostics else []
    gate_ok = bool(diagnostics and gate_labels and all(label in GATE_LABELS for label in gate_labels))
    claims.append(
        {
            "category": "gate contract",
            "claim": "The deployment gate emits exactly one allowed label for each evaluated N.",
            "status": _status(gate_ok, partial=bool(diagnostics)),
            "evidence_strength": "STRONG" if gate_ok else "WEAK",
            "evidence": "results/experiment_video_diagnostics.json; figures/figure3_video_diagnostics.png",
        }
    )

    occlusion_ok = bool(occlusion and occlusion.get("key_result", {}).get("occlusion_uncertainty_slope", 0.0) > 0.0)
    claims.append(
        {
            "category": "occlusion stress",
            "claim": "Occlusion stress increases uncertainty and exposes when visual selection is most vulnerable to hidden crossing artifacts.",
            "status": _status(occlusion_ok, partial=bool(occlusion)),
            "evidence_strength": "STRONG" if occlusion_ok else "WEAK",
            "evidence": "results/experiment_occlusion_stress.json; figures/figure4_occlusion_stress.png",
        }
    )

    final_status = _final_audit_status(root)
    final_ok = final_status in FINAL_AUDIT_STATUSES
    claims.append(
        {
            "category": "final audit",
            "claim": "The repository final audit chooses one allowed readiness status.",
            "status": _status(final_ok),
            "evidence_strength": "STRONG" if final_ok else "WEAK",
            "evidence": {"status": final_status, "allowed": FINAL_AUDIT_STATUSES},
        }
    )

    forbidden_hits = find_forbidden_claims(root)
    claims.append(
        {
            "category": "forbidden overclaims",
            "claim": "Universal video, robotics, and blanket top-score video overclaims are absent from README, docs, and paper text.",
            "status": "SUPPORTED" if not forbidden_hits else "UNSUPPORTED",
            "evidence_strength": "STRONG" if not forbidden_hits else "WEAK",
            "evidence": {"hits": forbidden_hits, "blocked_phrases": FORBIDDEN_CLAIMS},
        }
    )

    weak_count = len(weak_reasons) if full_multiseed else 0
    return {
        "schema_version": 1,
        "project": "When Plausible Videos Lie",
        "claims": claims,
        "forbidden_claims": FORBIDDEN_CLAIMS,
        "full_multiseed_evidence": full_multiseed,
        "weak_count": weak_count,
        "weak_reasons": weak_reasons,
        "all_supported_or_partial": all(c["status"] in {"SUPPORTED", "PARTIAL"} for c in claims),
        "unsupported_count": sum(1 for c in claims if c["status"] == "UNSUPPORTED"),
    }


def claim_status_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Claim Status",
        "",
        f"- Project: {payload['project']}",
        f"- Unsupported count: {payload['unsupported_count']}",
        f"- Weak strong-evidence checks: {payload.get('weak_count', 0)}",
        f"- Full multi-seed evidence: {payload.get('full_multiseed_evidence', False)}",
        "",
        "| Category | Status | Strength | Claim | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for claim in payload["claims"]:
        evidence = claim.get("evidence", "")
        if not isinstance(evidence, str):
            evidence = json.dumps(evidence, sort_keys=True)
        lines.append(
            f"| {claim['category']} | {claim['status']} | {claim.get('evidence_strength', 'NA')} | {claim['claim']} | {evidence} |"
        )
    if payload.get("weak_reasons"):
        lines.extend(["", "## Weak Evidence Reasons", ""])
        for reason in payload["weak_reasons"]:
            lines.append(f"- {reason}")
    lines.extend(["", "## Audit Block List", ""])
    for phrase in payload["forbidden_claims"]:
        lines.append(f"- `{phrase}`")
    lines.append("")
    return "\n".join(lines)
