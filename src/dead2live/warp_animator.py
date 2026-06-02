"""Universal renderer: animate ANY 2D portrait by warping its real pixels.

Unlike the procedural puppet (which redraws flat-art features), this deforms
the actual image with a Thin-Plate-Spline driven by control points, so it works
on photos, painted art, anime, etc. Feature motion (blink / mouth / brow /
gaze) is a TPS warp; head pose is a masked affine on top.

Honest limits: warping can squash an open eye toward closed but cannot synthesise
a real eyelid, and cannot reveal occluded teeth — for photoreal eye-close /
teeth you need the neural path (LivePortrait, Phase 2). Everything else
(smile, talk, surprise, brow, head turn/nod/tilt) is convincing.
"""
from __future__ import annotations

import numpy as np
import cv2

from .rig import Rig
from .animator import AnimationState, _c


class WarpAnimator:
    def __init__(self, image_bgr: np.ndarray, rig: Rig):
        self.img = image_bgr
        self.rig = rig
        self.h, self.w = image_bgr.shape[:2]
        self.bg = _c(image_bgr[2, 2])
        self._tps = cv2.createThinPlateSplineShapeTransformer() \
            if hasattr(cv2, "createThinPlateSplineShapeTransformer") else None
        self.src = self._build_src()                 # (N,2) float32 source pts
        self.head_mask = self._build_head_mask()

    # ----- control points -------------------------------------------------
    def _build_src(self) -> np.ndarray:
        w, h = self.w, self.h
        r = self.rig
        pts = [
            (0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
            (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2),
        ]
        self.idx = {"border": list(range(8))}
        le, re_ = r.left_eye, r.right_eye
        fcx = (le.cx + re_.cx) / 2
        fcy = (le.cy + re_.cy) / 2

        def add(name, p):
            self.idx[name] = len(pts)
            pts.append(p)

        # face-oval anchors (move only with head pose)
        hr = r.head_radius or (re_.cx - le.cx)
        add("face_top", (fcx, fcy - hr * 0.95))
        add("face_l", (fcx - hr, fcy))
        add("face_r", (fcx + hr, fcy))
        add("chin", r.head_pivot)
        add("nose", (fcx, (fcy + (r.mouth.cy if r.mouth else fcy + hr)) / 2))

        for tag, e in (("L", le), ("R", re_)):
            add(f"eye{tag}_c", (e.cx, e.cy))
            add(f"eye{tag}_t", (e.cx, e.cy - e.ry))
            add(f"eye{tag}_b", (e.cx, e.cy + e.ry))
            add(f"eye{tag}_i", (e.cx - e.rx, e.cy))
            add(f"eye{tag}_o", (e.cx + e.rx, e.cy))
        for tag, b in (("L", r.left_brow), ("R", r.right_brow)):
            if b:
                add(f"brow{tag}", (b.cx, b.cy))
        if r.mouth:
            m = r.mouth
            add("mouth_c", (m.cx, m.cy))
            add("mouth_l", (m.cx - m.w / 2, m.cy))
            add("mouth_r", (m.cx + m.w / 2, m.cy))
            add("mouth_t", (m.cx, m.cy - m.h / 2))
            add("mouth_b", (m.cx, m.cy + m.h / 2))
        return np.array(pts, np.float32)

    def _build_head_mask(self) -> np.ndarray:
        r = self.rig
        m = np.zeros((self.h, self.w), np.uint8)
        if r.face_box:
            fx, fy, fw, fh = r.face_box
            cx, cyc, ax, ay = fx + fw / 2, fy + fh / 2, fw * 0.62, fh * 0.60
        else:
            le, re_ = r.left_eye, r.right_eye
            fcx = (le.cx + re_.cx) / 2
            fcy = (le.cy + re_.cy) / 2
            hr = r.head_radius or (re_.cx - le.cx)
            chin_y = r.head_pivot[1]
            top_y = fcy - hr * 1.15
            cx, cyc, ax, ay = fcx, (top_y + chin_y) / 2, hr * 1.15, (chin_y - top_y) / 2
        cv2.ellipse(m, (int(cx), int(cyc)), (int(ax), int(ay)), 0, 0, 360, 255, -1)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=self.w * 0.02)
        return (m.astype(np.float32) / 255.0)[..., None]

    # ----- per-frame targets ----------------------------------------------
    def _targets(self, st: AnimationState) -> np.ndarray:
        dst = self.src.copy()
        r = self.rig
        for tag, e in (("L", r.left_eye), ("R", r.right_eye)):
            op = max(0.0, min(1.3, st.eye_open_l if tag == "L" else st.eye_open_r))
            # squash eyelids toward centre to blink; expand a bit to widen
            dst[self.idx[f"eye{tag}_t"]][1] = e.cy - e.ry * op
            dst[self.idx[f"eye{tag}_b"]][1] = e.cy + e.ry * op
            # gaze: nudge eye centre (drags iris pixels)
            dst[self.idx[f"eye{tag}_c"]][0] = e.cx + st.pupil_dx * e.rx * 0.45
            dst[self.idx[f"eye{tag}_c"]][1] = e.cy + st.pupil_dy * e.ry * 0.45
        for tag, b in (("L", r.left_brow), ("R", r.right_brow)):
            if b and f"brow{tag}" in self.idx:
                raise_px = st.brow_raise_l if tag == "L" else st.brow_raise_r
                ang = st.brow_angle * (1 if tag == "L" else -1)
                dst[self.idx[f"brow{tag}"]][1] = b.cy - raise_px + ang * 0.4
        if r.mouth and "mouth_c" in self.idx:
            m = r.mouth
            corner_up = st.mouth_smile * m.h * 1.4
            dst[self.idx["mouth_l"]][1] = m.cy - corner_up
            dst[self.idx["mouth_r"]][1] = m.cy - corner_up
            opening = st.mouth_open * max(m.h * 2.5, self.h * 0.03)
            dst[self.idx["mouth_t"]][1] = m.cy - m.h / 2 - opening * 0.4
            dst[self.idx["mouth_b"]][1] = m.cy + m.h / 2 + opening * 0.6
        return dst

    # ----- render ----------------------------------------------------------
    def render(self, st: AnimationState) -> np.ndarray:
        frame = self.img
        if self._tps is not None:
            dst = self._targets(st)
            if not np.allclose(dst, self.src):
                frame = self._warp(self.src, dst, self.img)
        return self._apply_head(frame, st)

    def _warp(self, src, dst, img):
        N = len(src)
        s = src.reshape(1, N, 2).astype(np.float32)
        d = dst.reshape(1, N, 2).astype(np.float32)
        matches = [cv2.DMatch(i, i, 0) for i in range(N)]
        # estimate mapping dst->src so warpImage pulls src pixels to dst positions
        self._tps.estimateTransformation(d, s, matches)
        try:
            return self._tps.warpImage(img, flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return img

    def _apply_head(self, frame, st: AnimationState) -> np.ndarray:
        if st.head_roll == 0 and st.head_yaw == 0 and st.head_pitch == 0:
            return frame
        px, py = self.rig.head_pivot
        M = cv2.getRotationMatrix2D((float(px), float(py)), st.head_roll, 1.0)
        M[0, 2] += st.head_yaw * self.w * 0.05
        M[1, 2] += st.head_pitch * self.h * 0.045
        moved = cv2.warpAffine(frame, M, (self.w, self.h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        m = self.head_mask
        return (frame.astype(np.float32) * (1 - m) + moved.astype(np.float32) * m).astype(np.uint8)


if __name__ == "__main__":
    from .detectors import MediaPipeRigDetector
    img = cv2.imread("assets/test_faces/real_face.jpg")
    rig = MediaPipeRigDetector().detect(img)
    a = WarpAnimator(img, rig)
    shots = {
        "neutral": AnimationState(),
        "smile": AnimationState(mouth_smile=1.0, mouth_open=0.12),
        "surprised": AnimationState(eye_open_l=1.25, eye_open_r=1.25,
                                    brow_raise_l=14, brow_raise_r=14, mouth_open=0.6),
        "blink": AnimationState(eye_open_l=0.05, eye_open_r=0.05),
        "talk": AnimationState(mouth_open=0.5),
        "turn": AnimationState(head_yaw=0.8, head_roll=6),
    }
    tiles = []
    for n, s in shots.items():
        f = a.render(s)
        cv2.putText(f, n, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        tiles.append(cv2.resize(f, (256, 256)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite("outputs/warp_photo.png", np.vstack(rows))
    print("saved outputs/warp_photo.png")
