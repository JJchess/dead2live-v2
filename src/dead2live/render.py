"""Render a timeline to frames and export mp4 / gif."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio

from .animator import PuppetAnimator, AnimationState


def render_frames(anim: PuppetAnimator, states: list[AnimationState]) -> list[np.ndarray]:
    """Render BGR frames for a list of states."""
    return [anim.render(s) for s in states]


def save_mp4(frames_bgr: list[np.ndarray], path: str | Path, fps: int = 25) -> str:
    path = str(path)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    # pad to even dims for h264
    h, w = rgb[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    rgb = [f[:h2, :w2] for f in rgb]
    imageio.mimsave(path, rgb, fps=fps, codec="libx264", quality=8,
                    macro_block_size=None)
    return path


def save_gif(frames_bgr: list[np.ndarray], path: str | Path, fps: int = 25,
             max_w: int = 360) -> str:
    path = str(path)
    out = []
    for f in frames_bgr:
        h, w = f.shape[:2]
        if w > max_w:
            f = cv2.resize(f, (max_w, int(h * max_w / w)))
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, out, duration=1.0 / fps, loop=0)
    return path
