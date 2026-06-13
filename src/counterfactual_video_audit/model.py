"""A compact CPU-feasible video transformer over frame patches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - tests skip model training if torch is absent.
    torch = None
    nn = None

from counterfactual_video_audit.candidates import actions_for_type, make_candidate
from counterfactual_video_audit.envs import GridVideoWorld, random_actions, valid_detour_actions


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the tiny video transformer")
    return torch


class TinyVideoTransformer(nn.Module if nn is not None else object):
    """Patch-wise autoregressive predictor for small RGB videos."""

    def __init__(
        self,
        frame_size: int = 16,
        patch_size: int = 4,
        channels: int = 3,
        d_model: int = 16,
        nhead: int = 2,
        num_layers: int = 1,
        max_frames: int = 12,
    ) -> None:
        _require_torch()
        super().__init__()
        self.frame_size = int(frame_size)
        self.patch_size = int(patch_size)
        self.channels = int(channels)
        self.patch_dim = channels * patch_size * patch_size
        self.patches_per_frame = (frame_size // patch_size) ** 2
        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.time_embed = nn.Embedding(max_frames, d_model)
        self.patch_embed = nn.Embedding(self.patches_per_frame, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, self.patch_dim)

    def patchify(self, video: Any) -> Any:
        b, t, c, h, w = video.shape
        p = self.patch_size
        video = video.reshape(b, t, c, h // p, p, w // p, p)
        video = video.permute(0, 1, 3, 5, 2, 4, 6).reshape(b, t, self.patches_per_frame, self.patch_dim)
        return video

    def forward(self, video_prefix: Any) -> Any:
        patches = self.patchify(video_prefix)
        b, t, p, _ = patches.shape
        device = patches.device
        x = self.patch_proj(patches)
        time_ids = torch.arange(t, device=device).view(1, t, 1)
        patch_ids = torch.arange(p, device=device).view(1, 1, p)
        x = x + self.time_embed(time_ids) + self.patch_embed(patch_ids)
        x = x.reshape(b, t * p, -1)
        encoded = self.encoder(x).reshape(b, t, p, -1)
        return self.head(encoded)


def downsample_video(video: np.ndarray, frame_size: int = 16) -> np.ndarray:
    if video.shape[1] == frame_size:
        return video.astype(np.float32)
    stride = max(1, video.shape[1] // frame_size)
    return video[:, ::stride, ::stride, :][:, :frame_size, :frame_size, :].astype(np.float32)


def make_training_videos(seed: int, count: int, frame_size: int = 16) -> np.ndarray:
    world = GridVideoWorld(frame_size=32)
    rng = np.random.default_rng(seed)
    videos = []
    for i in range(count):
        if i % 3 == 0:
            actions = valid_detour_actions(world.horizon)
        elif i % 3 == 1:
            actions = random_actions(rng, world.horizon)
        else:
            actions = actions_for_type(world, "near_miss", rng)
        candidate = make_candidate(world, "valid_detour" if i % 3 == 0 else "random", actions)
        videos.append(downsample_video(candidate.true_frames, frame_size=frame_size))
    return np.stack(videos, axis=0)


def train_tiny_video_transformer(
    output_path: Path,
    *,
    seed: int = 0,
    smoke: bool = False,
) -> dict[str, Any]:
    torch_mod = _require_torch()
    torch_mod.manual_seed(seed)
    np.random.seed(seed)
    frame_size = 16
    train_count = 6 if smoke else 24
    max_steps = 2 if smoke else 8
    videos = make_training_videos(seed, train_count, frame_size=frame_size)
    tensor = torch_mod.tensor(videos).permute(0, 1, 4, 2, 3).contiguous()
    model = TinyVideoTransformer(frame_size=frame_size)
    optimizer = torch_mod.optim.AdamW(model.parameters(), lr=1e-3)
    losses: list[float] = []
    batch_size = 6 if smoke else 8
    steps = 0
    while steps < max_steps:
        order = torch_mod.randperm(len(tensor))
        for start in range(0, len(tensor), batch_size):
            batch = tensor[order[start : start + batch_size]]
            prefix = batch[:, :-1]
            target = model.patchify(batch[:, 1:])
            pred = model(prefix)
            loss = torch_mod.mean((pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
            if steps >= max_steps:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch_mod.save(
        {
            "model": "TinyVideoTransformer",
            "config": {"frame_size": frame_size, "patch_size": 4, "d_model": 16, "max_frames": 12},
            "state_dict": model.state_dict(),
            "loss_history": losses,
            "smoke": bool(smoke),
            "seed": int(seed),
        },
        output_path,
    )
    return {
        "artifact": str(output_path),
        "train_count": train_count,
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_delta": losses[0] - losses[-1],
    }
