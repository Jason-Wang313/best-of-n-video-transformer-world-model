"""Train the tiny learned video transformer artifact."""

from __future__ import annotations

from experiments.common import RESULTS, write_json
from video_transformer_best_of_n.model import train_tiny_video_transformer


def run(*, smoke: bool = False, seed: int = 0) -> dict[str, object]:
    payload = train_tiny_video_transformer(
        RESULTS / "learned_tiny_video_transformer.pt",
        seed=seed,
        smoke=smoke,
    )
    payload["schema_version"] = 1
    payload["smoke"] = bool(smoke)
    write_json(RESULTS / "experiment_learned_video_transformer.json", payload)
    return payload


if __name__ == "__main__":
    run()
