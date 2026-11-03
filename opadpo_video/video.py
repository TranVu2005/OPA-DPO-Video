"""Video decoding and the distorted video m' used by the image-focus loss (Eq. 8)."""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
import torch


def _decode_decord(path: str, n_frames: int) -> tuple[np.ndarray, float]:
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0))
    total, fps = len(vr), vr.get_avg_fps() or 30.0
    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    return vr.get_batch(idx).asnumpy(), max(total / fps, 1e-3)


def _decode_av(path: str, n_frames: int) -> tuple[np.ndarray, float]:
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        duration = float(stream.duration * stream.time_base) if stream.duration else len(frames) / float(stream.average_rate or 30)
    idx = np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    return np.stack([frames[i] for i in idx]), max(duration, 1e-3)


@lru_cache(maxsize=64)
def load_video(path: str, n_frames: int) -> tuple[torch.Tensor, float]:
    """Uniformly sample `n_frames` frames. Returns uint8 (T, C, H, W) and the effective fps."""
    try:
        frames, duration = _decode_decord(path, n_frames)
    except ImportError:
        frames, duration = _decode_av(path, n_frames)
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return video, n_frames / duration


def sample_seed(sample_id: str) -> int:
    """Stable per-sample seed, so the reference pass and training see the same m'."""
    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "little")


def mask_video(video: torch.Tensor, ratio: float, method: str, seed: int) -> torch.Tensor:
    """Replace a `ratio` of the content with the per-frame mean colour (paper: 30% of pixels)."""
    g = torch.Generator().manual_seed(seed)
    out = video.clone()
    t, c, h, w = out.shape
    mean = out.float().mean(dim=(2, 3), keepdim=True).round().to(out.dtype)  # (T, C, 1, 1)
    if method == "frame":
        n = max(1, round(t * ratio))
        idx = torch.randperm(t, generator=g)[:n]
        out[idx] = mean[idx].expand(-1, -1, h, w)
        return out
    n = int(h * w * ratio)
    if method == "tube":
        pix = torch.randperm(h * w, generator=g)[:n]
        flat = out.view(t, c, h * w)
        flat[:, :, pix] = mean.view(t, c, 1).expand(-1, -1, n)
        return out
    if method == "pixel":
        flat = out.view(t, c, h * w)
        for i in range(t):
            pix = torch.randperm(h * w, generator=g)[:n]
            flat[i][:, pix] = mean[i].view(c, 1).expand(-1, n)
        return out
    raise ValueError(f"unknown mask method {method!r}")
