"""Aggregate multi-seed evidence for failure and repair claims."""

from __future__ import annotations

from experiments.common import RESULTS, TABLES, ci95, write_csv, write_json
from experiments.experiment_video_repairs import run as run_repairs
from video_transformer_best_of_n.envs import GridVideoWorld
from video_transformer_best_of_n.evaluation import evaluate_best_of_n, high_minus_low


def run(*, smoke: bool = False, seed: int = 0) -> dict[str, object]:
    seed_count = 1 if smoke else 3
    trials = 6 if smoke else 28
    rows = []
    failure_deltas = {
        "raw_plausibility_delta_high_n": [],
        "raw_action_violation_delta_high_n": [],
        "raw_real_utility_delta_high_n": [],
    }
    repair_improvements = []
    for idx in range(seed_count):
        s = seed + idx * 101
        _, summary = evaluate_best_of_n(trials=trials, seed=s, world=GridVideoWorld())
        plaus = high_minus_low(summary, "visual_plausibility")
        action = high_minus_low(summary, "action_consistency_violation_rate")
        utility = high_minus_low(summary, "real_utility")
        failure_deltas["raw_plausibility_delta_high_n"].append(plaus)
        failure_deltas["raw_action_violation_delta_high_n"].append(action)
        failure_deltas["raw_real_utility_delta_high_n"].append(utility)
        repair_payload = run_repairs(smoke=True, seed=s + 55, write_artifacts=False)
        repair_gain = float(repair_payload["key_result"]["calibrated_repair_n64_improvement_over_raw"])
        repair_improvements.append(repair_gain)
        rows.append(
            {
                "seed": s,
                "raw_plausibility_delta_high_n": plaus,
                "raw_action_violation_delta_high_n": action,
                "raw_real_utility_delta_high_n": utility,
                "calibrated_repair_n64_improvement_over_raw": repair_gain,
            }
        )

    failure = {}
    for key, values in failure_deltas.items():
        ci = ci95(values)
        failure[f"{key}_mean"] = ci["mean"]
        failure[f"{key}_lo"] = ci["lo"]
        failure[f"{key}_hi"] = ci["hi"]
    repair_ci = ci95(repair_improvements)
    repair = {
        "calibrated_repair_n64_improvement_over_raw_mean": repair_ci["mean"],
        "calibrated_repair_n64_improvement_over_raw_lo": repair_ci["lo"],
        "calibrated_repair_n64_improvement_over_raw_hi": repair_ci["hi"],
    }
    write_csv(TABLES / "multiseed_strong_evidence.csv", rows)
    payload = {
        "schema_version": 1,
        "smoke": bool(smoke),
        "seed": int(seed),
        "seed_count": seed_count,
        "rows": rows,
        "failure": failure,
        "repair": repair,
    }
    write_json(RESULTS / "multiseed_strong_evidence.json", payload)
    return payload


if __name__ == "__main__":
    run()
