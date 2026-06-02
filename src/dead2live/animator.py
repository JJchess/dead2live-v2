"""Puppet animator: render a portrait in any expression / pose.

Strategy for flat illustrations: erase the animatable features once to build a
clean "base face" (skin painted over eyes / brows / mouth), then *re-draw* those
features procedurally every frame from the rig's sampled palette, and finally
apply a small affine head transform. This is fully controllable, deterministic
and real-time on CPU - no neural model needed for the stylised domain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .rig import Rig, Eye, Brow, Mouth


# --------------------------------------------------------------------------- #
#  Animation state - the full controllable pose of the character at one instant
# --------------------------------------------------------------------------- #
@dataclass
class AnimationState:
    # eyes: 1 = fully open, 0 = closed
    eye_open_l: float = 1.0
    eye_open_r: float = 1.0
    # gaze: normalised pupil offset within the eye, -1..1
    pupil_dx: float = 0.0
    pupil_dy: float = 0.0
    # brows: raise (px-ish, +up), angle (deg, + = inner-down "angry")
    brow_raise_l: float = 0.0
    brow_raise_r: float = 0.0
    brow_angle: float = 0.0
    # mouth
    mouth_open: float = 0.0      # 0 closed .. 1 wide open
    mouth_smile: float = 0.2     # -1 frown .. 1 big smile
    mouth_width: float = 1.0     # horizontal scale
    # head pose (small)
    head_roll: float = 0.0       # degrees, + = tilt clockwise
    head_yaw: float = 0.0        # -1..1 left/right shift
    head_pitch: float = 0.0      # -1..1 up/down shift (+ = down/nod)

    def lerp(self, other: "AnimationState", t: float) -> "AnimationState":
        a, b = self.__dict__, other.__dict__
        return AnimationState(**{k: a[k] + (b[k] - a[k]) * t for k in a})


def _c(bgr) -> tuple:
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _shade(bgr, f: float) -> tuple:
    """Multiply brightness by f (f<1 darker)."""
    return tuple(int(max(0, min(255, v * f))) for v in bgr)


# --------------------------------------------------------------------------- #
class PuppetAnimator:
    def __init__(self, image_bgr: np.ndarray, rig: Rig):
        self.img = image_bgr
        self.rig = rig
        self.bg = _c(image_bgr[2, 2])           # corner = background fill
        self.base = self._build_base()
        self.head_mask = self._build_head_mask()   # float32 HxWx1, feathered 0..1

    def _build_head_mask(self) -> np.ndarray:
        """Soft elliptical mask over hair+face down to the chin, so head motion
        moves only the head and leaves the body / background still."""
        h, w = self.img.shape[:2]
        r = self.rig
        if r.left_eye and r.right_eye:
            cx = (r.left_eye.cx + r.right_eye.cx) / 2
            eye_y = (r.left_eye.cy + r.right_eye.cy) / 2
        else:
            cx, eye_y = w / 2, h * 0.37
        # chin a little below the mouth; top above the hair
        chin_y = (r.mouth.cy + r.mouth.h * 1.4) if r.mouth else h * 0.66
        top_y = max(0.0, eye_y - h * 0.30)
        cyc = (top_y + chin_y) / 2
        ax = w * 0.37
        ay = (chin_y - top_y) / 2
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (int(cx), int(cyc)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=w * 0.02)   # feather edges
        return (m.astype(np.float32) / 255.0)[..., None]

    # ----- build clean base (features erased to skin) ---------------------
    def _build_base(self) -> np.ndarray:
        base = self.img.copy()
        skin = _c(self.rig.skin)
        r = self.rig
        for eye in (r.left_eye, r.right_eye):
            if eye:
                cv2.ellipse(base, (int(eye.cx), int(eye.cy)),
                            (int(eye.rx * 1.18), int(eye.ry * 1.18)),
                            0, 0, 360, skin, -1)
        for brow in (r.left_brow, r.right_brow):
            if brow:
                cv2.ellipse(base, (int(brow.cx), int(brow.cy)),
                            (int(brow.w * 0.75), int(brow.h * 0.95)),
                            0, 0, 360, skin, -1)
        if r.mouth:
            m = r.mouth
            cv2.ellipse(base, (int(m.cx), int(m.cy)),
                        (int(m.w * 0.62), int(m.h * 1.15)),
                        0, 0, 360, skin, -1)
        return base

    # ----- render one frame ----------------------------------------------
    def render(self, st: AnimationState) -> np.ndarray:
        frame = self.base.copy()
        self._draw_brow(frame, self.rig.left_brow, st.brow_raise_l, +st.brow_angle)
        self._draw_brow(frame, self.rig.right_brow, st.brow_raise_r, -st.brow_angle)
        self._draw_eye(frame, self.rig.left_eye, st.eye_open_l, st)
        self._draw_eye(frame, self.rig.right_eye, st.eye_open_r, st)
        self._draw_mouth(frame, self.rig.mouth, st)
        frame = self._apply_head(frame, st)
        return frame

    # ---------------------------------------------------------------------
    def _draw_eye(self, frame, eye: Eye, openness: float, st: AnimationState):
        if eye is None:
            return
        cx, cy = int(eye.cx), int(eye.cy)
        rx, ry = eye.rx, eye.ry
        openness = max(0.0, min(1.0, openness))
        if openness < 0.12:
            # closed: a gentle eyelid crease
            cv2.ellipse(frame, (cx, cy), (int(rx * 0.95), max(2, int(ry * 0.18))),
                        0, 180, 360, _shade(self.rig.skin, 0.72), max(2, int(ry * 0.12)))
            return
        ry_o = max(2.0, ry * openness)
        # eye white + pupil drawn on a layer, clipped to the (squashed) white
        layer = frame.copy()
        cv2.ellipse(layer, (cx, cy), (int(rx), int(ry_o)), 0, 0, 360, _c(eye.white), -1)
        px = cx + int(st.pupil_dx * (rx - eye.pupil_r) * 0.9)
        py = cy + int(st.pupil_dy * (ry_o - eye.pupil_r * 0.5) * 0.9)
        cv2.circle(layer, (px, py), int(eye.pupil_r), _c(eye.pupil), -1)
        # tiny catch-light for liveliness
        cv2.circle(layer, (px - int(eye.pupil_r * 0.3), py - int(eye.pupil_r * 0.3)),
                   max(1, int(eye.pupil_r * 0.22)), (245, 245, 245), -1)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.ellipse(mask, (cx, cy), (int(rx), int(ry_o)), 0, 0, 360, 255, -1)
        frame[mask > 0] = layer[mask > 0]

    def _draw_brow(self, frame, brow: Brow, raise_px: float, angle: float):
        if brow is None:
            return
        cx = int(brow.cx)
        cy = int(brow.cy - raise_px)
        cv2.ellipse(frame, (cx, cy), (int(brow.w * 0.5), max(2, int(brow.h * 0.5))),
                    angle, 0, 360, _c(brow.color), -1)

    def _draw_mouth(self, frame, mouth: Mouth, st: AnimationState):
        if mouth is None:
            return
        cx, cy = int(mouth.cx), int(mouth.cy)
        w = mouth.w * st.mouth_width
        line = _c(mouth.color)
        thick = max(2, int(mouth.h * 0.45))
        if st.mouth_open > 0.12:
            # open mouth: filled cavity + lips outline
            oh = int(mouth.h * (0.4 + st.mouth_open * 2.0))
            ow = int(w * 0.5 * (0.8 + st.mouth_open * 0.25))
            oy = cy + oh // 5
            cv2.ellipse(frame, (cx, oy), (ow, oh), 0, 0, 360, _c(mouth.inner), -1)
            cv2.ellipse(frame, (cx, oy), (ow, oh), 0, 0, 360, line, max(2, thick // 2))
            return
        # closed: smile / frown arc.  smile>0 -> U (start 0 end 180 below centre)
        s = max(-1.0, min(1.0, st.mouth_smile))
        depth = int(abs(s) * mouth.h * 1.6) + 2
        ax = int(w * 0.5)
        if s >= 0:
            cv2.ellipse(frame, (cx, cy - depth // 2), (ax, depth), 0, 20, 160, line, thick)
        else:
            cv2.ellipse(frame, (cx, cy + depth // 2), (ax, depth), 0, 200, 340, line, thick)

    def _apply_head(self, frame, st: AnimationState) -> np.ndarray:
        if st.head_roll == 0 and st.head_yaw == 0 and st.head_pitch == 0:
            return frame
        h, w = frame.shape[:2]
        px, py = self.rig.head_pivot
        M = cv2.getRotationMatrix2D((float(px), float(py)), st.head_roll, 1.0)
        M[0, 2] += st.head_yaw * w * 0.05
        M[1, 2] += st.head_pitch * h * 0.045
        moved = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        # composite the transformed head over the still body via the feathered
        # head footprint -> nod/shake/tilt move only the head, not the frame
        m = self.head_mask
        out = frame.astype(np.float32) * (1 - m) + moved.astype(np.float32) * m
        return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .rig import CartoonRigDetector
    image = cv2.imread("test_image.png")
    rig = CartoonRigDetector().detect(image)
    anim = PuppetAnimator(image, rig)

    demos = {
        "neutral":   AnimationState(),
        "blink":     AnimationState(eye_open_l=0.05, eye_open_r=0.05),
        "smile":     AnimationState(mouth_smile=1.0, mouth_open=0.15),
        "surprised": AnimationState(eye_open_l=1.3, eye_open_r=1.3, brow_raise_l=18,
                                    brow_raise_r=18, mouth_open=0.7, mouth_smile=0),
        "angry":     AnimationState(brow_angle=22, brow_raise_l=-4, brow_raise_r=-4,
                                    mouth_smile=-0.6, eye_open_l=0.8, eye_open_r=0.8),
        "sad":       AnimationState(brow_angle=-16, mouth_smile=-0.8,
                                    eye_open_l=0.7, eye_open_r=0.7, head_pitch=0.3),
        "look_left": AnimationState(pupil_dx=-0.9),
        "talk":      AnimationState(mouth_open=0.5, mouth_smile=0.1),
        "tilt":      AnimationState(head_roll=12, mouth_smile=0.6),
    }
    tiles = []
    for name, st in demos.items():
        f = anim.render(st)
        cv2.putText(f, name, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        tiles.append(cv2.resize(f, (300, 318)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite("outputs/expressions_grid.png", np.vstack(rows))
    print("saved outputs/expressions_grid.png")
