"""Small Moving-MNIST benchmark stress test for video-future selection."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

MNIST_IMAGES_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
CANVAS_SIZE = 64
DIGIT_SIZE = 28
CONTEXT = 10
FUTURE = 10
MAX_N = 64


@dataclass(frozen=True)
class MovingMnistSequence:
    """One generated 10-to-10 Moving-MNIST-style evaluation case."""

    digits: tuple[np.ndarray, np.ndarray]
    positions: np.ndarray
    frames: np.ndarray
    context_velocity: np.ndarray
    future_bounces: int


@dataclass(frozen=True)
class MovingMnistCandidate:
    """One predicted future for a Moving-MNIST sequence."""

    kind: str
    positions: np.ndarray
    frames: np.ndarray
    visual_score: float
    future_mse: float
    centroid_ade: float
    speed_ratio: float
    motion_collapse: float
    boundary_violation: float


def load_digit_bank(cache_dir: Path, *, limit: int = 1024, allow_download: bool = True) -> tuple[np.ndarray, str]:
    """Load MNIST digits, falling back to deterministic templates for offline tests."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "train-images-idx3-ubyte.gz"
    if allow_download and not path.exists():
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(MNIST_IMAGES_URL, tmp)
        tmp.replace(path)
    if path.exists():
        with gzip.open(path, "rb") as handle:
            magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
            if magic != 2051:
                raise ValueError(f"unexpected MNIST magic number {magic}")
            read_count = min(int(limit), int(count))
            raw = handle.read(read_count * rows * cols)
        digits = np.frombuffer(raw, dtype=np.uint8).reshape(read_count, rows, cols).astype(np.float32) / 255.0
        return _resize_digits(digits), "mnist_train_images"
    return _template_digits(max(16, min(limit, 128))), "template_fallback"


def _resize_digits(digits: np.ndarray) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = []
    for digit in digits:
        image = Image.fromarray(np.asarray(255.0 * digit, dtype=np.uint8), mode="L")
        image = image.resize((DIGIT_SIZE, DIGIT_SIZE), resampling)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if float(arr.max()) > 0.05:
            resized.append(arr)
    return np.asarray(resized, dtype=np.float32)


def _template_digits(count: int) -> np.ndarray:
    font_digits = []
    for idx in range(int(count)):
        image = Image.new("L", (28, 28), 0)
        draw = ImageDraw.Draw(image)
        draw.text((7, 4), str(idx % 10), fill=255)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        font_digits.append(arr)
    return _resize_digits(np.asarray(font_digits, dtype=np.float32))


def render_digits(digits: tuple[np.ndarray, np.ndarray], positions: np.ndarray) -> np.ndarray:
    frames = []
    for pos_pair in np.asarray(positions, dtype=float):
        frame = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
        for digit, pos in zip(digits, pos_pair):
            _paste_digit(frame, digit, pos)
        frames.append(np.clip(frame, 0.0, 1.0))
    return np.asarray(frames, dtype=np.float32)


def _paste_digit(frame: np.ndarray, digit: np.ndarray, pos: np.ndarray) -> None:
    x = int(round(float(pos[0])))
    y = int(round(float(pos[1])))
    h, w = digit.shape
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(CANVAS_SIZE, x + w)
    y1 = min(CANVAS_SIZE, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    dx0 = x0 - x
    dy0 = y0 - y
    patch = digit[dy0 : dy0 + (y1 - y0), dx0 : dx0 + (x1 - x0)]
    frame[y0:y1, x0:x1] = np.maximum(frame[y0:y1, x0:x1], patch)


def _step_bounce(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    limit = CANVAS_SIZE - DIGIT_SIZE
    next_pos = pos + vel
    next_vel = vel.copy()
    bounces = 0
    for digit_idx in range(next_pos.shape[0]):
        for axis in range(2):
            if next_pos[digit_idx, axis] < 0 or next_pos[digit_idx, axis] > limit:
                next_vel[digit_idx, axis] *= -1.0
                next_pos[digit_idx, axis] = pos[digit_idx, axis] + next_vel[digit_idx, axis]
                bounces += 1
    return np.clip(next_pos, 0.0, float(limit)), next_vel, bounces


def _simulate_bounce(start_pos: np.ndarray, start_vel: np.ndarray, steps: int) -> tuple[np.ndarray, int]:
    positions = [np.asarray(start_pos, dtype=float).copy()]
    pos = np.asarray(start_pos, dtype=float).copy()
    vel = np.asarray(start_vel, dtype=float).copy()
    bounce_count = 0
    for _ in range(int(steps) - 1):
        pos, vel, bounces = _step_bounce(pos, vel)
        bounce_count += bounces
        positions.append(pos.copy())
    return np.asarray(positions, dtype=float), bounce_count


def make_sequences(digits: np.ndarray, *, count: int, seed: int) -> list[MovingMnistSequence]:
    rng = np.random.default_rng(seed)
    velocities = np.asarray(
        [
            [[3.0, 1.0], [-2.0, 2.0]],
            [[-3.0, 1.0], [2.0, -2.0]],
            [[2.0, 3.0], [-2.0, -2.0]],
            [[1.0, -3.0], [-3.0, 2.0]],
            [[3.0, -2.0], [2.0, 2.0]],
        ],
        dtype=float,
    )
    sequences: list[MovingMnistSequence] = []
    attempts = 0
    while len(sequences) < int(count) and attempts < int(count) * 80:
        attempts += 1
        selected = digits[rng.choice(len(digits), size=2, replace=False)]
        vel = velocities[int(rng.integers(0, len(velocities)))].copy()
        limit = CANVAS_SIZE - DIGIT_SIZE
        start = rng.uniform(4.0, limit - 4.0, size=(2, 2))
        # Bias one digit toward a future bounce after the ten context frames.
        axis = int(rng.integers(0, 2))
        start[0, axis] = rng.choice([rng.uniform(2.0, 8.0), rng.uniform(limit - 8.0, limit - 2.0)])
        if start[0, axis] < limit / 2:
            vel[0, axis] = abs(vel[0, axis])
        else:
            vel[0, axis] = -abs(vel[0, axis])
        positions, total_bounces = _simulate_bounce(start, vel, CONTEXT + FUTURE)
        future_bounces = _future_bounce_count(positions)
        if future_bounces < 1 and total_bounces < 1:
            continue
        context_velocity = positions[CONTEXT - 1] - positions[CONTEXT - 2]
        frames = render_digits((selected[0], selected[1]), positions)
        sequences.append(
            MovingMnistSequence(
                digits=(selected[0], selected[1]),
                positions=positions,
                frames=frames,
                context_velocity=context_velocity,
                future_bounces=future_bounces,
            )
        )
    if len(sequences) < int(count):
        raise RuntimeError("could not generate enough bouncing Moving-MNIST sequences")
    return sequences


def _future_bounce_count(positions: np.ndarray) -> int:
    future = positions[CONTEXT - 1 :]
    velocities = np.diff(future, axis=0)
    signs = np.sign(velocities)
    flips = signs[1:] * signs[:-1] < 0
    return int(np.sum(flips))


def baseline_positions(sequence: MovingMnistSequence, kind: str) -> np.ndarray:
    last = sequence.positions[CONTEXT - 1].copy()
    vel = sequence.context_velocity.copy()
    if kind == "persistence":
        return np.repeat(last[None, :, :], FUTURE, axis=0)
    if kind == "constant_velocity_bounce":
        future, _ = _simulate_bounce(last, vel, FUTURE + 1)
        return future[1:]
    if kind == "constant_velocity_linear":
        return np.asarray([last + vel * (step + 1) for step in range(FUTURE)], dtype=float)
    raise ValueError(f"unknown baseline kind {kind}")


def sample_candidate(sequence: MovingMnistSequence, rng: np.random.Generator) -> np.ndarray:
    kind = str(rng.choice(["persistence", "slow", "linear", "bounce", "noisy_bounce"], p=[0.18, 0.30, 0.18, 0.24, 0.10]))
    last = sequence.positions[CONTEXT - 1].copy()
    vel = sequence.context_velocity.copy()
    if kind == "persistence":
        positions = baseline_positions(sequence, "persistence")
    elif kind == "slow":
        positions = np.asarray([last + 0.28 * vel * (step + 1) for step in range(FUTURE)], dtype=float)
    elif kind == "linear":
        positions = baseline_positions(sequence, "constant_velocity_linear")
    elif kind == "bounce":
        positions = baseline_positions(sequence, "constant_velocity_bounce")
    elif kind == "noisy_bounce":
        positions = baseline_positions(sequence, "constant_velocity_bounce")
        positions = positions + rng.normal(0.0, 1.35, size=positions.shape)
    else:
        raise ValueError(kind)
    return positions, kind


def make_candidate(sequence: MovingMnistSequence, positions: np.ndarray, kind: str) -> MovingMnistCandidate:
    future_true = sequence.frames[CONTEXT : CONTEXT + FUTURE]
    true_positions = sequence.positions[CONTEXT : CONTEXT + FUTURE]
    frames = render_digits(sequence.digits, positions)
    mse = float(np.mean((frames - future_true) ** 2))
    centroid = float(np.mean(np.linalg.norm(np.asarray(positions, dtype=float) - true_positions, axis=2)))
    score, speed_ratio, collapse, boundary = visual_future_score(sequence, positions, frames)
    return MovingMnistCandidate(
        kind=kind,
        positions=np.asarray(positions, dtype=float),
        frames=frames,
        visual_score=score,
        future_mse=mse,
        centroid_ade=centroid,
        speed_ratio=speed_ratio,
        motion_collapse=collapse,
        boundary_violation=boundary,
    )


def visual_future_score(sequence: MovingMnistSequence, positions: np.ndarray, frames: np.ndarray) -> tuple[float, float, float, float]:
    context_last = sequence.frames[CONTEXT - 1]
    stacked = np.concatenate([context_last[None, :, :], frames], axis=0)
    frame_delta = float(np.mean(np.abs(np.diff(stacked, axis=0))))
    flicker_score = 1.0 - min(1.0, 18.0 * frame_delta)
    context_mass = float(np.mean(sequence.frames[:CONTEXT]))
    future_mass = float(np.mean(frames))
    mass_score = 1.0 - min(1.0, abs(future_mass - context_mass) / max(context_mass, 1e-6))
    sharpness = float(np.mean([np.std(frame) for frame in frames]))
    sharpness_score = min(1.0, 9.0 * sharpness)
    speed = float(np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=2)))
    context_speed = float(np.mean(np.linalg.norm(sequence.context_velocity, axis=1)))
    speed_ratio = speed / max(context_speed, 1e-6)
    collapse = 1.0 if speed_ratio < 0.35 else 0.0
    limit = CANVAS_SIZE - DIGIT_SIZE
    boundary = float(np.mean((positions < 0.0) | (positions > limit)))
    boundary_score = 1.0 - min(1.0, 3.0 * boundary)
    score = 0.58 * flicker_score + 0.20 * mass_score + 0.12 * sharpness_score + 0.10 * boundary_score
    return float(np.clip(score, 0.0, 1.0)), float(speed_ratio), collapse, boundary


def candidate_pool(sequence: MovingMnistSequence, *, seed: int, n: int = MAX_N) -> list[MovingMnistCandidate]:
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(int(n)):
        positions, kind = sample_candidate(sequence, rng)
        pool.append(make_candidate(sequence, positions, kind))
    return pool


def select_raw(pool: list[MovingMnistCandidate]) -> MovingMnistCandidate:
    return max(pool, key=lambda item: item.visual_score)


def select_motion_gate(sequence: MovingMnistSequence, pool: list[MovingMnistCandidate]) -> MovingMnistCandidate:
    kept = [item for item in pool if item.motion_collapse < 0.5 and item.boundary_violation <= 0.02 and item.speed_ratio <= 1.35]
    if kept:
        return max(kept, key=lambda item: item.visual_score)
    return make_candidate(sequence, baseline_positions(sequence, "constant_velocity_bounce"), "constant_velocity_bounce")


def select_oracle(pool: list[MovingMnistCandidate]) -> MovingMnistCandidate:
    return min(pool, key=lambda item: item.future_mse)


def run_moving_mnist_benchmark(
    *,
    cache_dir: Path,
    sequence_count: int = 48,
    seed: int = 2026,
    allow_download: bool = True,
) -> tuple[list[dict[str, float | int | str]], dict[str, str | int]]:
    digits, source = load_digit_bank(cache_dir, allow_download=allow_download)
    sequences = make_sequences(digits, count=sequence_count, seed=seed)
    rows: list[dict[str, float | int | str]] = []
    n_values = (1, 4, 16, 64)
    for seq_idx, sequence in enumerate(sequences):
        pool = candidate_pool(sequence, seed=seed * 1000 + seq_idx, n=max(n_values))
        baselines = {
            "persistence": make_candidate(sequence, baseline_positions(sequence, "persistence"), "persistence"),
            "constant_velocity_bounce": make_candidate(
                sequence, baseline_positions(sequence, "constant_velocity_bounce"), "constant_velocity_bounce"
            ),
        }
        for n in n_values:
            subpool = pool[:n]
            selected = {
                "raw_visual": select_raw(subpool),
                "motion_gate": select_motion_gate(sequence, subpool),
                "oracle": select_oracle(subpool),
                **baselines,
            }
            for selector, candidate in selected.items():
                rows.append(
                    {
                        "sequence": seq_idx,
                        "N": n,
                        "selector": selector,
                        "candidate_kind": candidate.kind,
                        "visual_score": candidate.visual_score,
                        "future_mse": candidate.future_mse,
                        "future_psnr": _psnr(candidate.future_mse),
                        "centroid_ade": candidate.centroid_ade,
                        "speed_ratio": candidate.speed_ratio,
                        "motion_collapse": candidate.motion_collapse,
                        "boundary_violation": candidate.boundary_violation,
                        "future_bounces": sequence.future_bounces,
                    }
                )
    meta = {
        "dataset": "Moving MNIST 10-to-10 future prediction subset",
        "digit_source": source,
        "sequence_count": int(sequence_count),
        "seed": int(seed),
        "context_frames": CONTEXT,
        "future_frames": FUTURE,
        "canvas_size": CANVAS_SIZE,
    }
    return rows, meta


def summarize_benchmark_rows(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    summary = []
    selectors = sorted({str(row["selector"]) for row in rows})
    n_values = sorted({int(row["N"]) for row in rows})
    for selector in selectors:
        for n in n_values:
            group = [row for row in rows if str(row["selector"]) == selector and int(row["N"]) == n]
            if not group:
                continue
            item: dict[str, float | int | str] = {"selector": selector, "N": n}
            for key in [
                "visual_score",
                "future_mse",
                "future_psnr",
                "centroid_ade",
                "speed_ratio",
                "motion_collapse",
                "boundary_violation",
            ]:
                item[key] = float(np.mean([float(row[key]) for row in group]))
            kinds = {str(row["candidate_kind"]) for row in group}
            for kind in sorted(kinds):
                item[f"kind_rate_{kind}"] = float(np.mean([str(row["candidate_kind"]) == kind for row in group]))
            summary.append(item)
    return summary


def benchmark_claims(summary: list[dict[str, float | int | str]]) -> dict[str, object]:
    def row(selector: str, n: int) -> dict[str, float | int | str]:
        matches = [item for item in summary if str(item["selector"]) == selector and int(item["N"]) == n]
        if not matches:
            raise ValueError(f"missing summary row for {selector=} {n=}")
        return matches[0]

    raw_1 = row("raw_visual", 1)
    raw_64 = row("raw_visual", 64)
    gate_64 = row("motion_gate", 64)
    persistence_64 = row("persistence", 64)
    bounce_64 = row("constant_velocity_bounce", 64)
    oracle_64 = row("oracle", 64)
    visual_gain = float(raw_64["visual_score"]) - float(raw_1["visual_score"])
    mse_change = float(raw_64["future_mse"]) - float(raw_1["future_mse"])
    ade_change = float(raw_64["centroid_ade"]) - float(raw_1["centroid_ade"])
    gate_mse_repair = float(raw_64["future_mse"]) - float(gate_64["future_mse"])
    gate_ade_repair = float(raw_64["centroid_ade"]) - float(gate_64["centroid_ade"])
    baseline_mse_gap = float(persistence_64["future_mse"]) - float(bounce_64["future_mse"])
    oracle_gap = float(raw_64["future_mse"]) - float(oracle_64["future_mse"])

    claims = {
        "moving_mnist_raw_visual_extremizes": _claim(
            visual_gain > 0.10,
            visual_gain,
            0.10,
            "Increasing candidate count to 64 raises the raw internal visual score on Moving MNIST.",
        ),
        "moving_mnist_raw_mse_worsens": _claim(
            mse_change > 0.004,
            mse_change,
            0.004,
            "The same high-candidate raw selection worsens future-frame MSE.",
        ),
        "moving_mnist_raw_centroid_ade_worsens": _claim(
            ade_change > 2.0,
            ade_change,
            2.0,
            "The same high-candidate raw selection worsens digit-centroid displacement error.",
        ),
        "moving_mnist_motion_gate_repairs_mse": _claim(
            gate_mse_repair > 0.005,
            gate_mse_repair,
            0.005,
            "The motion-support gate reduces high-candidate future-frame MSE relative to raw visual selection.",
        ),
        "moving_mnist_motion_gate_repairs_ade": _claim(
            gate_ade_repair > 3.0,
            gate_ade_repair,
            3.0,
            "The motion-support gate reduces high-candidate centroid error relative to raw visual selection.",
        ),
        "moving_mnist_bounce_baseline_beats_persistence": _claim(
            baseline_mse_gap > 0.008,
            baseline_mse_gap,
            0.008,
            "A standard constant-velocity bounce baseline beats last-frame persistence on future MSE.",
        ),
        "moving_mnist_oracle_gap_visible": _claim(
            oracle_gap > 0.008,
            oracle_gap,
            0.008,
            "The candidate pool contains better futures than raw visual selection chooses.",
        ),
    }
    return {
        "all_passed": all(item["status"] == "pass" for item in claims.values()),
        "claims": claims,
        "summary": (
            f"visual gain {visual_gain:.3f}, MSE change {mse_change:.4f}, ADE change {ade_change:.3f}, "
            f"gate MSE repair {gate_mse_repair:.4f}, gate ADE repair {gate_ade_repair:.3f}."
        ),
    }


def _claim(status: bool, value: float, threshold: float, description: str) -> dict[str, float | str]:
    return {
        "status": "pass" if status else "fail",
        "value": float(value),
        "threshold": float(threshold),
        "description": description,
    }


def _psnr(mse: float) -> float:
    return float(99.0 if mse <= 1e-12 else -10.0 * np.log10(float(mse)))
