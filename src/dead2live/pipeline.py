"""Auto-routing: given ANY 2D portrait, pick the right detector + renderer.

  flat illustration  -> CartoonRigDetector / MediaPipe / Gemini  + PuppetAnimator
  photo / painted art -> MediaPipe / Gemini                      + WarpAnimator

Detectors are tried cheapest-and-most-precise first; the first whose geometry
validates well enough wins (so we only call the Gemini API when offline
detectors fail). Everything downstream (the Brain, the timeline) is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rig import Rig, CartoonRigDetector
from .detectors import MediaPipeRigDetector, GeminiVisionRigDetector
from .animator import PuppetAnimator
from .warp_animator import WarpAnimator
from .interfaces import rig_score, is_flat_art

# cache detector instances (MediaPipe keeps a loaded model)
_cartoon = CartoonRigDetector(); _cartoon.name = "cartoon-cv"
_mediapipe = MediaPipeRigDetector()
_gemini = GeminiVisionRigDetector()

GOOD_ENOUGH = 0.8       # stop trying further detectors at/above this score
USABLE = 0.35           # below this we consider detection failed


@dataclass
class PipelineInfo:
    flat: bool
    detector: str
    renderer: str
    score: float
    rig: Rig


def build_animator(image_bgr: np.ndarray, prefer_warp: Optional[bool] = None):
    """Return (animator, PipelineInfo). Raises if no detector finds a face."""
    flat = is_flat_art(image_bgr) if prefer_warp is None else (not prefer_warp)
    order = [_cartoon, _mediapipe, _gemini] if flat else [_mediapipe, _gemini]

    best = None  # (score, detector_name, rig)
    for det in order:
        try:
            rig = det.detect(image_bgr)
        except Exception:
            rig = None
        sc = rig_score(rig)
        if best is None or sc > best[0]:
            best = (sc, det.name, rig)
        if sc >= GOOD_ENOUGH:
            break

    if best is None or best[2] is None or best[0] < USABLE:
        raise ValueError("No face detected by any backend "
                         f"(best score={best[0] if best else 0:.2f}). "
                         "Try a clearer, front-facing portrait.")

    score, det_name, rig = best
    AnimCls = PuppetAnimator if flat else WarpAnimator
    anim = AnimCls(image_bgr, rig)
    info = PipelineInfo(flat=flat, detector=det_name,
                        renderer=AnimCls.__name__, score=score, rig=rig)
    return anim, info


if __name__ == "__main__":
    import cv2, numpy as np
    from .rig import draw_rig_overlay
    tests = ["test_image.png", "assets/test_faces/anime_girl.png",
             "assets/test_faces/blue_mascot.png", "assets/test_faces/old_man.png",
             "assets/test_faces/real_face.jpg"]
    from . import brain, render
    brain.llm_parse_script = None
    tiles = []
    for p in tests:
        im = cv2.imread(p)
        try:
            anim, info = build_animator(im)
            states, _ = brain.animate("开心地笑", 25)
            f = anim.render(states[len(states) // 2])
            tag = f"{info.detector}/{info.renderer[:4]} {info.score:.2f}"
        except Exception as e:
            f = im.copy(); tag = f"ERR {type(e).__name__}"
        f = cv2.resize(f, (256, 256))
        cv2.putText(f, p.split("/")[-1][:10], (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(f, tag, (6, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        tiles.append(f)
        print(f"{p.split('/')[-1]:18} -> {tag}")
    cv2.imwrite("outputs/universality.png", np.hstack(tiles))
    print("saved outputs/universality.png")
