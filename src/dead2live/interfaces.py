"""Pluggable contracts that make dead2live general across any 2D character.

Two stable interfaces decouple the domain-specific parts from the universal
ones:

  RigDetector.detect(image_bgr) -> Rig | None
      How to *locate* facial features on an image. Different backends cover
      different domains (flat art / photos / arbitrary styles) but all emit the
      same ``Rig``.

  Animator.render(state: AnimationState) -> frame_bgr
      How to *render* the character in a given pose. Procedural redraw (flat
      art) or mesh warp (anything) — both consume the same ``AnimationState``.

The Brain (text -> AnimationState timeline) sits entirely above these and is
therefore character-agnostic by construction.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .rig import Rig
from .animator import AnimationState


@runtime_checkable
class RigDetector(Protocol):
    name: str
    def detect(self, image_bgr: np.ndarray) -> Optional[Rig]: ...


@runtime_checkable
class Animator(Protocol):
    def render(self, state: AnimationState) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
#  Geometry validation - lets the router reject a bad detection and fall back
# --------------------------------------------------------------------------- #
def rig_score(rig: Optional[Rig]) -> float:
    """0 = unusable, 1 = textbook face. Used to pick the best detector.
    Supports single-eye profile rigs (one eye is allowed to be None)."""
    if rig is None or not rig.eyes:
        return 0.0
    w, h = rig.width, rig.height

    # --- profile / single visible eye ---
    if len(rig.eyes) == 1:
        e = rig.eyes[0]
        s = 0.9
        if not (0.03 * h < e.cy < 0.82 * h):
            s *= 0.5
        if not (0.05 * w < e.cx < 0.95 * w):
            s *= 0.5
        if rig.mouth is not None and rig.mouth.cy < e.cy - e.ry:
            s *= 0.5
        return float(max(0.0, min(1.0, s)))

    # --- frontal / two eyes ---
    le, re_ = rig.left_eye, rig.right_eye
    s = 1.0

    # eyes must be left/right of each other with a sane horizontal gap
    dx = abs(le.cx - re_.cx)
    if not (0.06 * w < dx < 0.6 * w):
        return 0.0
    # eyes roughly level
    dy = abs(le.cy - re_.cy)
    if dy > 0.12 * h:
        s *= 0.5
    # eyes in the upper-middle of the image
    if not (0.05 * h < (le.cy + re_.cy) / 2 < 0.7 * h):
        s *= 0.5
    # similar eye sizes
    ratio = (le.rx + 1) / (re_.rx + 1)
    if ratio < 0.5 or ratio > 2.0:
        s *= 0.6
    # mouth below the eyes
    if rig.mouth is not None:
        if rig.mouth.cy > (le.cy + re_.cy) / 2:
            s *= 1.0
        else:
            s *= 0.5
    else:
        s *= 0.7
    return float(max(0.0, min(1.0, s)))


def is_flat_art(image_bgr: np.ndarray, sample: int = 200) -> bool:
    """Heuristic: flat illustrations have few distinct colours and large solid
    regions; photos/painted art have smooth gradients and many colours.

    Returns True for flat / vector / cartoon art (-> procedural renderer)."""
    img = image_bgr
    h, w = img.shape[:2]
    scale = sample / max(h, w)
    small = img if scale >= 1 else __import__("cv2").resize(img, (int(w * scale), int(h * scale)))
    # quantise to 5 bits/channel and count unique colours
    q = (small.astype(np.int32) >> 3)
    codes = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    uniq = np.unique(codes).size
    n = codes.size
    # fraction of pixels belonging to the top-8 most common colours
    vals, counts = np.unique(codes, return_counts=True)
    top = np.sort(counts)[::-1][:8].sum() / n
    # flat art: few colours AND dominated by a handful of solids
    return (uniq < n * 0.04) and (top > 0.5)
