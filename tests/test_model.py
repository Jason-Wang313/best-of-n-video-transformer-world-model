from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_tiny_video_transformer_forward_shape():
    import torch

    from counterfactual_video_audit.model import TinyVideoTransformer

    model = TinyVideoTransformer(frame_size=16, patch_size=4, max_frames=12)
    video = torch.zeros(2, 4, 3, 16, 16)
    out = model(video)
    assert out.shape == (2, 4, 16, 48)
