"""Rig detectors for different image domains. All emit the same ``Rig``.

  CartoonRigDetector     - flat high-contrast illustrations           (rig.py)
  MediaPipeRigDetector   - real / semi-real photographed faces
  GeminiVisionRigDetector- ANY 2D style (anime, painting, mascot, ...) via a
                           multimodal VLM that returns feature coordinates

The router in ``pipeline.py`` tries them in order and keeps the first whose
geometry validates (see ``interfaces.rig_score``).
"""
from __future__ import annotations

import base64
import json
import urllib.request
from typing import Optional

import cv2
import numpy as np

from . import config
from .rig import Rig, Eye, Brow, Mouth, CartoonRigDetector, _median_color


# --------------------------------------------------------------------------- #
#  MediaPipe (photos)
# --------------------------------------------------------------------------- #
class MediaPipeRigDetector:
    name = "mediapipe"

    # canonical FaceMesh indices
    EYE_R = dict(outer=33, inner=133, top=159, bot=145)    # image-left eye
    EYE_L = dict(outer=263, inner=362, top=386, bot=374)   # image-right eye
    BROW_R = [70, 63, 105, 66, 107]
    BROW_L = [336, 296, 334, 293, 300]
    MOUTH = dict(l=61, r=291, top=13, bot=14)
    FACE = dict(top=10, chin=152, left=234, right=454)
    CHEEKS = [50, 280, 117, 346]

    def __init__(self):
        self._fm = None

    def _mesh(self):
        if self._fm is None:
            import mediapipe as mp
            self._fm = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=0.3)
        return self._fm

    def detect(self, image_bgr: np.ndarray) -> Optional[Rig]:
        h, w = image_bgr.shape[:2]
        res = self._mesh().process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        P = lambda i: (lm[i].x * w, lm[i].y * h)

        def eye(idx, iris_c, iris_pts):
            o, inn, t, b = P(idx["outer"]), P(idx["inner"]), P(idx["top"]), P(idx["bot"])
            cx = (o[0] + inn[0] + t[0] + b[0]) / 4
            cy = (o[1] + inn[1] + t[1] + b[1]) / 4
            rx = abs(o[0] - inn[0]) / 2 or w * 0.04
            ry = abs(t[1] - b[1]) / 2 or rx * 0.5
            ic = P(iris_c)
            ir = np.mean([np.hypot(P(p)[0] - ic[0], P(p)[1] - ic[1]) for p in iris_pts]) \
                if iris_pts else rx * 0.5
            white = _median_color(image_bgr, int((ic[0] + inn[0]) / 2), int(cy), 3)
            pup = _median_color(image_bgr, int(ic[0]), int(ic[1]), max(2, int(ir * 0.4)))
            return Eye(cx=cx, cy=cy, rx=max(rx, ry * 0.6), ry=max(ry, 3),
                       pupil_r=max(ir, 3), white=white, pupil=pup)

        has_iris = len(lm) >= 478
        eyeR = eye(self.EYE_R, 468 if has_iris else self.EYE_R["inner"],
                   [469, 470, 471, 472] if has_iris else [])
        eyeL = eye(self.EYE_L, 473 if has_iris else self.EYE_L["inner"],
                   [474, 475, 476, 477] if has_iris else [])

        def brow(ids):
            pts = np.array([P(i) for i in ids])
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            bw = pts[:, 0].max() - pts[:, 0].min()
            col = _median_color(image_bgr, int(cx), int(cy), 3)
            return Brow(cx=cx, cy=cy, w=max(bw, w * 0.05), h=max(bw * 0.18, 4), color=col)

        ml, mr = P(self.MOUTH["l"]), P(self.MOUTH["r"])
        mt, mb = P(self.MOUTH["top"]), P(self.MOUTH["bot"])
        mcx, mcy = (ml[0] + mr[0]) / 2, (mt[1] + mb[1]) / 2
        mouth = Mouth(cx=mcx, cy=mcy, w=abs(mr[0] - ml[0]), h=max(abs(mb[1] - mt[1]), 6),
                      color=_median_color(image_bgr, int(mcx), int(mt[1]), 2),
                      inner=[40, 50, 90])

        chin = P(self.FACE["chin"])
        skins = [_median_color(image_bgr, int(P(i)[0]), int(P(i)[1]), 4) for i in self.CHEEKS]
        skin = [int(v) for v in np.median(np.array(skins), axis=0)]

        rig = Rig(width=w, height=h, skin=skin,
                  left_eye=eyeR if eyeR.cx < eyeL.cx else eyeL,
                  right_eye=eyeL if eyeR.cx < eyeL.cx else eyeR,
                  left_brow=brow(self.BROW_R), right_brow=brow(self.BROW_L),
                  mouth=mouth, head_pivot=(chin[0], chin[1] + h * 0.04),
                  head_radius=abs(P(self.FACE["right"])[0] - P(self.FACE["left"])[0]) / 2,
                  source="mediapipe")
        return rig


# --------------------------------------------------------------------------- #
#  Gemini Vision (any 2D style)
# --------------------------------------------------------------------------- #
_VISION_PROMPT = """You are a precise facial-keypoint locator for a 2D character
(any style: anime, cartoon, painting, mascot, photo). Look at the image and
return NORMALIZED coordinates (0..1, x by width, y by height) of the face parts.
rx/rw are fractions of image WIDTH, ry/rh fractions of image HEIGHT.
"left_eye"/"right_eye" mean the eye on the LEFT/RIGHT side of the IMAGE.
Return ONLY JSON for the schema. If a part is missing, estimate from anatomy."""

_VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "left_eye":  {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "rx": {"type": "number"}, "ry": {"type": "number"},
            "pupil_r": {"type": "number"}}, "required": ["x", "y", "rx", "ry"]},
        "right_eye": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "rx": {"type": "number"}, "ry": {"type": "number"},
            "pupil_r": {"type": "number"}}, "required": ["x", "y", "rx", "ry"]},
        "left_brow":  {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}},
            "required": ["x", "y"]},
        "right_brow": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}},
            "required": ["x", "y"]},
        "mouth": {"type": "object", "properties": {
            "x": {"type": "number"}, "y": {"type": "number"},
            "w": {"type": "number"}, "h": {"type": "number"}},
            "required": ["x", "y", "w", "h"]},
        "face": {"type": "object", "properties": {
            "cx": {"type": "number"}, "cy": {"type": "number"},
            "rx": {"type": "number"}, "ry": {"type": "number"}},
            "required": ["cx", "cy", "rx", "ry"]},
    },
    "required": ["left_eye", "right_eye", "mouth", "face"],
}

_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent?key={key}")


class GeminiVisionRigDetector:
    name = "gemini-vision"

    def detect(self, image_bgr: np.ndarray) -> Optional[Rig]:
        if not config.GEMINI_API_KEY:
            return None
        h, w = image_bgr.shape[:2]
        # downscale for the API
        scale = 512 / max(h, w)
        small = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1 else image_bgr
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()

        url = _ENDPOINT.format(model=config.GEMINI_MODEL, key=config.GEMINI_API_KEY)
        body = {
            "contents": [{"role": "user", "parts": [
                {"text": _VISION_PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
            "generationConfig": {"temperature": 0.0,
                                 "response_mime_type": "application/json",
                                 "response_schema": _VISION_SCHEMA},
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=40) as r:
            resp = json.load(r)
        raw = "".join(p.get("text", "") for p in resp["candidates"][0]["content"]["parts"])
        d = json.loads(raw)
        return self._to_rig(d, image_bgr)

    def _to_rig(self, d: dict, img: np.ndarray) -> Rig:
        h, w = img.shape[:2]

        def mk_eye(e):
            cx, cy = e["x"] * w, e["y"] * h
            rx, ry = max(e["rx"] * w, 4), max(e["ry"] * h, 3)
            pr = e.get("pupil_r", e["ry"] * 0.6) * h
            off = int(rx * 0.32)
            cl = _median_color(img, int(cx - off), int(cy), 3)
            cr = _median_color(img, int(cx + off), int(cy), 3)
            white = cl if sum(cl) >= sum(cr) else cr
            pup = _median_color(img, int(cx), int(cy), max(2, int(pr * 0.4)))
            return Eye(cx=cx, cy=cy, rx=rx, ry=ry, pupil_r=max(pr, 3),
                       white=white, pupil=pup)

        def mk_brow(b, eye):
            if not b:
                return Brow(cx=eye.cx, cy=eye.cy - eye.ry * 2.2, w=eye.rx * 2,
                            h=max(eye.ry * 0.4, 4), color=[60, 60, 60])
            cx, cy = b["x"] * w, b["y"] * h
            bw = b.get("w", 0.12) * w
            return Brow(cx=cx, cy=cy, w=max(bw, eye.rx * 1.4),
                        h=max(bw * 0.18, 4),
                        color=_median_color(img, int(cx), int(cy), 3))

        le, re_ = mk_eye(d["left_eye"]), mk_eye(d["right_eye"])
        if le.cx > re_.cx:
            le, re_ = re_, le
        m = d["mouth"]
        mcx, mcy = m["x"] * w, m["y"] * h
        mw, mh = max(m["w"] * w, 8), max(m["h"] * h, 6)
        x0, x1 = int(mcx - mw / 2), int(mcx + mw / 2)
        y0, y1 = int(mcy - mh / 2), int(mcy + mh / 2)
        patch = img[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)].reshape(-1, 3)
        mcol = [int(v) for v in patch[int(np.argmin(patch.sum(axis=1)))]] if patch.size else [60, 60, 60]
        mouth = Mouth(cx=mcx, cy=mcy, w=mw, h=mh, color=mcol, inner=[40, 50, 90])

        f = d["face"]
        # skin: median of points around the face centre avoiding features
        pts = [(f["cx"], (le.cy / h + f["cy"]) / 2), (f["cx"] - f["rx"] * 0.4, f["cy"]),
               (f["cx"] + f["rx"] * 0.4, f["cy"])]
        skins = [_median_color(img, int(px * w), int(py * h), 4) for px, py in pts]
        skin = [int(v) for v in np.median(np.array(skins), axis=0)]

        pivot = (f["cx"] * w, (f["cy"] + f["ry"]) * h)
        return Rig(width=w, height=h, skin=skin, left_eye=le, right_eye=re_,
                   left_brow=mk_brow(d.get("left_brow"), le),
                   right_brow=mk_brow(d.get("right_brow"), re_),
                   mouth=mouth, head_pivot=pivot, head_radius=f["rx"] * w,
                   source="gemini-vision")


if __name__ == "__main__":
    from .interfaces import rig_score
    tests = ["test_image.png", "assets/test_faces/anime_girl.png",
             "assets/test_faces/blue_mascot.png", "assets/test_faces/old_man.png",
             "assets/test_faces/real_face.jpg"]
    mp_det, gv_det = MediaPipeRigDetector(), GeminiVisionRigDetector()
    for p in tests:
        im = cv2.imread(p)
        for det in (mp_det, gv_det):
            try:
                rig = det.detect(im)
                print(f"{p.split('/')[-1]:18} {det.name:14} score={rig_score(rig):.2f}")
            except Exception as e:
                print(f"{p.split('/')[-1]:18} {det.name:14} ERR {type(e).__name__}: {str(e)[:80]}")
