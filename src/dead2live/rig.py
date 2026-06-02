"""Rig detection: turn an arbitrary portrait into a controllable feature rig.

A *rig* is a plain dict describing where the animatable features live (eyes,
pupils, brows, mouth, head pivot) plus a sampled colour palette. It is the
contract between detection and the animator, so it can be produced by:

  * ``CartoonRigDetector`` - classical CV, robust on flat / illustrated faces
    (the test Shakespeare avatar, emoji, stickers, anime-ish art).
  * a MediaPipe FaceMesh path for photoreal faces (added later).
  * hand editing / a vision model writing the JSON directly.

All coordinates are absolute pixels in the source image. Colours are BGR
(OpenCV convention) stored as length-3 lists so the rig round-trips to JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Rig data model
# --------------------------------------------------------------------------- #
@dataclass
class Eye:
    cx: float
    cy: float
    rx: float          # white half-width
    ry: float          # white half-height
    pupil_r: float     # pupil radius
    white: list        # BGR
    pupil: list        # BGR


@dataclass
class Brow:
    cx: float
    cy: float
    w: float
    h: float
    color: list        # BGR


@dataclass
class Mouth:
    cx: float
    cy: float
    w: float
    h: float
    color: list        # BGR  (line / lip colour)
    inner: list        # BGR  (open-mouth interior colour)


@dataclass
class Rig:
    width: int
    height: int
    skin: list                       # BGR sampled face skin
    left_eye: Optional[Eye] = None
    right_eye: Optional[Eye] = None
    left_brow: Optional[Brow] = None
    right_brow: Optional[Brow] = None
    mouth: Optional[Mouth] = None
    head_pivot: tuple = (0.0, 0.0)   # (x,y) pivot for nod/tilt (neck)
    head_radius: float = 0.0
    source: str = "cartoon-cv"

    # ----- serialisation ---------------------------------------------------
    def to_json(self, path: str | Path) -> None:
        def enc(o):
            if isinstance(o, (Eye, Brow, Mouth)):
                return asdict(o)
            return o
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2, default=enc), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "Rig":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, cls in [("left_eye", Eye), ("right_eye", Eye),
                       ("left_brow", Brow), ("right_brow", Brow),
                       ("mouth", Mouth)]:
            if d.get(k):
                d[k] = cls(**d[k])
        d["head_pivot"] = tuple(d["head_pivot"])
        return Rig(**d)


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _components(mask: np.ndarray):
    """Return list of (area, x, y, w, h, cx, cy) for a uint8 mask, big->small."""
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        out.append((a, x, y, w, h, float(cent[i][0]), float(cent[i][1])))
    out.sort(reverse=True)
    return out


def _median_color(img: np.ndarray, cx: int, cy: int, r: int) -> list:
    h, w = img.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    patch = img[y0:y1, x0:x1].reshape(-1, 3)
    return [int(v) for v in np.median(patch, axis=0)]


# --------------------------------------------------------------------------- #
#  Cartoon / flat-illustration detector
# --------------------------------------------------------------------------- #
class CartoonRigDetector:
    """Detect facial features on flat, high-contrast illustrated portraits."""

    def detect(self, image_bgr: np.ndarray) -> Rig:
        img = image_bgr
        h, w = img.shape[:2]
        b, g, r = [c.astype(int) for c in cv2.split(img)]

        # --- skin: sample the face centre, robust to dark/white features ----
        skin = self._sample_skin(img)

        # --- eyes: near-white blobs, paired & symmetric in upper-mid face ---
        white = (((b > 195) & (g > 195) & (r > 195)).astype(np.uint8)) * 255
        eyes = self._find_eyes(white, img, w, h)

        # --- pupils: dark blob inside each eye white ------------------------
        dark = (((b < 95) & (g < 95) & (r < 95)).astype(np.uint8)) * 255
        if eyes:
            for eye in eyes:
                self._fit_pupil(eye, dark, img)

        left_eye = right_eye = None
        if len(eyes) == 2:
            eyes.sort(key=lambda e: e.cx)
            left_eye, right_eye = eyes[0], eyes[1]   # image-left, image-right

        # --- brows: thin dark blobs just above each eye ---------------------
        left_brow = right_brow = None
        if left_eye and right_eye:
            left_brow = self._find_brow(dark, left_eye, img)
            right_brow = self._find_brow(dark, right_eye, img)

        # --- mouth: dark/coloured region in lower-centre --------------------
        mouth = self._find_mouth(img, dark, w, h, eyes)

        # --- head pivot: neck, below mouth ----------------------------------
        cx = (left_eye.cx + right_eye.cx) / 2 if (left_eye and right_eye) else w / 2
        pivot = (cx, h * 0.78)
        head_r = w * 0.30

        return Rig(width=w, height=h, skin=skin,
                   left_eye=left_eye, right_eye=right_eye,
                   left_brow=left_brow, right_brow=right_brow,
                   mouth=mouth, head_pivot=pivot, head_radius=head_r)

    # ---------------------------------------------------------------------- #
    def _sample_skin(self, img) -> list:
        h, w = img.shape[:2]
        # candidate skin patches around the nose / cheeks (avoid eyes & mouth)
        pts = [(0.50, 0.45), (0.42, 0.42), (0.58, 0.42), (0.50, 0.30)]
        cols = []
        for fx, fy in pts:
            cols.append(_median_color(img, int(w * fx), int(h * fy), int(w * 0.04)))
        cols = np.array(cols)
        # pick the patch closest to the overall median (reject outliers)
        med = np.median(cols, axis=0)
        d = np.linalg.norm(cols - med, axis=1)
        return [int(v) for v in cols[int(np.argmin(d))]]

    def _find_eyes(self, white, img, w, h):
        comps = _components(white)
        cand = []
        for a, x, y, cw, ch, cx, cy in comps:
            if a < 0.0004 * w * h or a > 0.05 * w * h:
                continue
            if not (0.12 * h < cy < 0.62 * h):       # upper-middle band
                continue
            ar = cw / max(ch, 1)
            if ar < 0.4 or ar > 2.6:
                continue
            cand.append((a, x, y, cw, ch, cx, cy))
        # find the best symmetric horizontal pair
        best = None
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                A, B = cand[i], cand[j]
                if abs(A[6] - B[6]) > 0.06 * h:        # similar y
                    continue
                if abs(A[0] - B[0]) > 0.6 * max(A[0], B[0]):  # similar size
                    continue
                dx = abs(A[5] - B[5])
                if not (0.08 * w < dx < 0.5 * w):
                    continue
                score = abs(A[6] - B[6]) + abs(A[0] - B[0]) / max(A[0], B[0]) * 50
                if best is None or score < best[0]:
                    best = (score, A, B)
        eyes = []
        if best:
            for c in (best[1], best[2]):
                a, x, y, cw, ch, cx, cy = c
                # sample the WHITE ring (offset from centre, which holds the pupil)
                off = int(cw * 0.32)
                cl = _median_color(img, int(cx - off), int(cy), 3)
                cr = _median_color(img, int(cx + off), int(cy), 3)
                wc = cl if sum(cl) >= sum(cr) else cr
                eyes.append(Eye(cx=cx, cy=cy, rx=cw / 2, ry=ch / 2,
                                pupil_r=min(cw, ch) * 0.28, white=wc, pupil=[40, 40, 40]))
        return eyes

    def _fit_pupil(self, eye: Eye, dark, img):
        x0, x1 = int(eye.cx - eye.rx), int(eye.cx + eye.rx)
        y0, y1 = int(eye.cy - eye.ry), int(eye.cy + eye.ry)
        sub = dark[max(0, y0):y1, max(0, x0):x1]
        if sub.size == 0:
            return
        comps = _components(sub)
        if comps:
            a, x, y, cw, ch, cx, cy = comps[0]
            eye.pupil_r = max(cw, ch) / 2
            eye.cx_pupil = x0 + cx  # type: ignore[attr-defined]
            eye.pupil = _median_color(img, int(x0 + cx), int(y0 + cy), max(2, int(eye.pupil_r * 0.4)))

    def _find_brow(self, dark, eye: Eye, img) -> Optional[Brow]:
        # search band just above the eye white
        x0, x1 = int(eye.cx - eye.rx * 1.4), int(eye.cx + eye.rx * 1.4)
        y1 = int(eye.cy - eye.ry * 1.05)
        y0 = int(eye.cy - eye.ry * 3.2)
        if y0 < 0:
            y0 = 0
        sub = dark[y0:max(y0 + 1, y1), max(0, x0):x1]
        if sub.size == 0:
            return None
        comps = _components(sub)
        for a, x, y, cw, ch, cx, cy in comps:
            if cw < eye.rx * 0.4:        # too small/narrow to be a brow
                continue
            col = _median_color(img, int(max(0, x0) + cx), int(y0 + cy), 3)
            return Brow(cx=max(0, x0) + cx, cy=y0 + cy, w=cw, h=max(ch, 4), color=col)
        return None

    def _find_mouth(self, img, dark, w, h, eyes) -> Optional[Mouth]:
        # lower-centre band; prefer dark blob, fall back to a default
        cx = (eyes[0].cx + eyes[1].cx) / 2 if len(eyes) == 2 else w / 2
        y0, y1 = int(h * 0.52), int(h * 0.74)
        band = np.zeros_like(dark)
        band[y0:y1, int(w * 0.30):int(w * 0.70)] = dark[y0:y1, int(w * 0.30):int(w * 0.70)]
        comps = _components(band)
        expected_y = h * 0.62
        cands = []
        for a, x, y, cw, ch, mcx, mcy in comps:
            if a < 0.0004 * w * h:
                continue
            if cw < ch * 1.3:       # mouths are clearly wider than tall
                continue
            cands.append((abs(mcy - expected_y), cw, ch, mcx, mcy))
        if cands:
            # nearest to expected mouth line -> picks the lips/smile, not the moustache
            cands.sort()
            _, cw, ch, mcx, mcy = cands[0]
            # line colour = darkest pixel in the lip bbox (the thin smile stroke)
            x0, x1 = int(mcx - cw / 2), int(mcx + cw / 2)
            y0, y1 = int(mcy - ch / 2), int(mcy + ch / 2)
            patch = img[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
            col = [int(v) for v in patch[int(np.argmin(patch.sum(axis=1)))]]
            return Mouth(cx=mcx, cy=mcy, w=cw, h=max(ch, 6), color=col, inner=[40, 50, 90])
        # fallback default mouth
        return Mouth(cx=cx, cy=h * 0.62, w=w * 0.16, h=h * 0.04,
                     color=[60, 60, 60], inner=[40, 50, 90])


# --------------------------------------------------------------------------- #
#  Debug overlay
# --------------------------------------------------------------------------- #
def draw_rig_overlay(img: np.ndarray, rig: Rig) -> np.ndarray:
    o = img.copy()
    for eye, c in [(rig.left_eye, (0, 255, 0)), (rig.right_eye, (0, 255, 0))]:
        if eye:
            cv2.ellipse(o, (int(eye.cx), int(eye.cy)), (int(eye.rx), int(eye.ry)),
                        0, 0, 360, c, 2)
            cv2.circle(o, (int(eye.cx), int(eye.cy)), int(eye.pupil_r), (255, 0, 0), 2)
    for brow in [rig.left_brow, rig.right_brow]:
        if brow:
            cv2.rectangle(o, (int(brow.cx - brow.w / 2), int(brow.cy - brow.h / 2)),
                          (int(brow.cx + brow.w / 2), int(brow.cy + brow.h / 2)),
                          (0, 200, 255), 2)
    if rig.mouth:
        m = rig.mouth
        cv2.rectangle(o, (int(m.cx - m.w / 2), int(m.cy - m.h / 2)),
                      (int(m.cx + m.w / 2), int(m.cy + m.h / 2)), (255, 0, 255), 2)
    cv2.circle(o, (int(rig.head_pivot[0]), int(rig.head_pivot[1])), 5, (0, 0, 255), -1)
    return o


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
    image = cv2.imread(src)
    rig = CartoonRigDetector().detect(image)
    rig.to_json("assets/rigs/test_rig.json")
    cv2.imwrite("outputs/rig_overlay.png", draw_rig_overlay(image, rig))
    print("skin", rig.skin)
    print("left_eye", rig.left_eye)
    print("right_eye", rig.right_eye)
    print("left_brow", rig.left_brow, "right_brow", rig.right_brow)
    print("mouth", rig.mouth)
    print("saved outputs/rig_overlay.png + assets/rigs/test_rig.json")
