"""The "brain": turn a text instruction into an animation timeline.

   instruction text -> Plan(base expression + actions + duration)
                    -> list[AnimationState] (one per frame)

Expressions are *held poses*; actions are *time-varying motions* (blink, nod,
shake, talk, gaze) layered on top. An always-on idle track (auto-blink + a
breathing sway) keeps the character alive even for a bare expression.

Rule-based and bilingual (Chinese + English) by default; an optional LLM hook
(`llm_parse`) can be plugged in for free-form understanding - it only needs to
return the same Plan structure.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .animator import AnimationState


# --------------------------------------------------------------------------- #
#  Expression library (held target poses)
# --------------------------------------------------------------------------- #
EXPRESSIONS: dict[str, AnimationState] = {
    "neutral":   AnimationState(mouth_smile=0.25),
    "happy":     AnimationState(mouth_smile=1.0, mouth_open=0.12,
                                eye_open_l=0.92, eye_open_r=0.92),
    "sad":       AnimationState(brow_angle=-15, mouth_smile=-0.8,
                                eye_open_l=0.72, eye_open_r=0.72, head_pitch=0.28),
    "angry":     AnimationState(brow_angle=22, brow_raise_l=-5, brow_raise_r=-5,
                                mouth_smile=-0.6, eye_open_l=0.82, eye_open_r=0.82),
    "surprised": AnimationState(eye_open_l=1.35, eye_open_r=1.35, brow_raise_l=20,
                                brow_raise_r=20, mouth_open=0.6, mouth_smile=0.0),
    "fear":      AnimationState(eye_open_l=1.25, eye_open_r=1.25, brow_raise_l=14,
                                brow_raise_r=14, brow_angle=-8, mouth_open=0.3,
                                mouth_smile=-0.4),
    "thinking":  AnimationState(brow_raise_l=8, brow_raise_r=-2, mouth_smile=-0.1,
                                pupil_dx=0.5, pupil_dy=-0.5, head_roll=5),
    "wink":      AnimationState(eye_open_l=0.05, mouth_smile=0.9),
    # 无语 / 无奈 / 翻白眼：半眯眼 + 眼球上翻 + 嘴角平/微下 + 轻微歪头（面瘫脸）
    "speechless": AnimationState(eye_open_l=0.55, eye_open_r=0.55, pupil_dy=-0.65,
                                 brow_raise_l=7, brow_raise_r=7, mouth_smile=-0.2,
                                 head_roll=4),
    "disgust":   AnimationState(brow_angle=10, brow_raise_l=4, mouth_smile=-0.5,
                                eye_open_l=0.7, eye_open_r=0.7, pupil_dy=-0.2),
}

# emotion -> expression aliases (bilingual keywords)
_EXPR_KEYWORDS = {
    "happy":     ["happy", "smile", "joy", "glad", "微笑", "笑", "开心", "高兴", "快乐", "喜悦"],
    "sad":       ["sad", "unhappy", "cry", "难过", "伤心", "悲伤", "沮丧", "哭"],
    "angry":     ["angry", "mad", "furious", "rage", "生气", "愤怒", "恼火", "发火", "气愤"],
    "surprised": ["surprised", "shock", "astonish", "wow", "惊讶", "吃惊", "震惊", "惊喜", "意外"],
    "fear":      ["fear", "scared", "afraid", "害怕", "恐惧", "惊恐"],
    "thinking":  ["think", "ponder", "hmm", "思考", "想", "沉思", "疑惑"],
    "neutral":   ["neutral", "calm", "normal", "平静", "正常", "中性", "放松"],
    "wink":      ["wink", "眨眼睛", "使眼色", "抛媚眼"],
    "speechless": ["speechless", "eye roll", "whatever", "无语", "无奈", "翻白眼",
                   "醉了", "服了", "尴尬", "一言难尽", "面瘫"],
    "disgust":   ["disgust", "disgusted", "yuck", "嫌弃", "恶心", "鄙视", "嫌弃脸"],
}

# action keywords
_ACTION_KEYWORDS = {
    "blink":  ["blink", "眨眼", "眨"],
    "nod":    ["nod", "yes", "agree", "点头", "同意", "嗯"],
    "shake":  ["shake head", "shake", "no", "disagree", "摇头", "不同意", "拒绝"],
    "talk":   ["talk", "speak", "say", "speech", "说话", "讲话", "说", "讲", "念", "朗读"],
    "look_left":  ["look left", "向左", "看左", "左看"],
    "look_right": ["look right", "向右", "看右", "右看"],
    "look_up":    ["look up", "向上", "看上", "抬头看"],
    "look_down":  ["look down", "向下", "看下", "低头看"],
    "yawn":       ["yawn", "哈欠", "打哈欠", "好困", "犯困"],
    "laugh":      ["laugh", "大笑", "哈哈", "笑出声", "狂笑"],
}

_INTENSE_UP = ["very", "extremely", "super", "so ", "非常", "很", "超级", "特别", "极其"]
_INTENSE_DN = ["slightly", "a little", "a bit", "稍微", "有点", "略微", "轻轻"]


# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    base: str = "neutral"
    intensity: float = 1.0
    actions: list[str] = field(default_factory=list)
    duration: float = 3.0
    speak_text: Optional[str] = None    # text whose length drives talk length
    seed: int = 7


# --------------------------------------------------------------------------- #
#  Parser
# --------------------------------------------------------------------------- #
def parse(text: str) -> Plan:
    t = text.lower().strip()
    plan = Plan()

    # expression (last match wins -> most specific)
    for expr, kws in _EXPR_KEYWORDS.items():
        if any(k in t for k in kws):
            plan.base = expr

    # actions (multiple allowed)
    for act, kws in _ACTION_KEYWORDS.items():
        if any(k in t for k in kws):
            plan.actions.append(act)

    # intensity
    if any(k in t for k in _INTENSE_UP):
        plan.intensity = 1.45
    if any(k in t for k in _INTENSE_DN):
        plan.intensity = 0.6

    # talking: pull quoted text or text after 说/say to time the mouth
    m = re.search(r"[\"'“”‘’](.+?)[\"'“”‘’]", text)
    if m:
        plan.speak_text = m.group(1)
        if "talk" not in plan.actions:
            plan.actions.append("talk")
    if "talk" in plan.actions:
        n = len(plan.speak_text or text)
        plan.duration = max(3.0, min(12.0, 1.2 + n * 0.12))

    if "nod" in plan.actions or "shake" in plan.actions:
        plan.duration = max(plan.duration, 2.6)

    # deterministic-but-varied seed from the text
    plan.seed = abs(hash(text)) % 100000
    return plan


# optional LLM hooks (set by brain_gemini.enable()):
#   llm_parse(text)        -> Plan                (single beat)
#   llm_parse_script(text) -> list[Plan]          (multi-beat sequence)
llm_parse: Optional[Callable[[str], Plan]] = None
llm_parse_script: Optional[Callable[[str], list]] = None


def make_plan(text: str) -> Plan:
    if llm_parse is not None:
        try:
            return llm_parse(text)
        except Exception:
            pass
    return parse(text)


def make_script(text: str) -> list[Plan]:
    """Return a sequence of beats. LLM may split a compound instruction into
    several beats; rule-based path returns a single beat."""
    if llm_parse_script is not None:
        try:
            s = llm_parse_script(text)
            if s:
                return s
        except Exception:
            pass
    return [make_plan(text)]


# --------------------------------------------------------------------------- #
#  Easing
# --------------------------------------------------------------------------- #
def _smooth(x: float) -> float:           # smoothstep 0..1
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def _scale_expr(st: AnimationState, k: float) -> AnimationState:
    """Scale an expression's deviation from neutral by k."""
    base = AnimationState()
    d = {}
    for key, v in st.__dict__.items():
        nv = base.__dict__[key]
        d[key] = nv + (v - nv) * k
    return AnimationState(**d)


# --------------------------------------------------------------------------- #
#  Timeline builder
# --------------------------------------------------------------------------- #
def build_timeline(plan: Plan, fps: int = 25,
                   start_state: AnimationState | None = None) -> list[AnimationState]:
    rng = random.Random(plan.seed)
    n = max(1, int(plan.duration * fps))
    target = _scale_expr(EXPRESSIONS.get(plan.base, EXPRESSIONS["neutral"]), plan.intensity)
    start = start_state if start_state is not None else AnimationState()

    # schedule auto-blinks (skip if a closed-eye expression/action governs eyes)
    blink_times = []
    if plan.base != "wink":
        t = rng.uniform(0.6, 1.6)
        while t < plan.duration - 0.2:
            blink_times.append(t)
            t += rng.uniform(1.8, 3.6)
    if "blink" in plan.actions:
        blink_times.append(0.5)

    # talk mouth envelope (deterministic pseudo-speech)
    talk = "talk" in plan.actions
    talk_phon = [rng.uniform(0.15, 0.85) for _ in range(int(plan.duration * 7) + 2)]

    frames: list[AnimationState] = []
    for i in range(n):
        t = i / fps
        st = start.lerp(target, _smooth(t / 0.35))          # ease from prev pose

        # ---- idle life: breathing sway + micro head motion ----
        breathe = math.sin(t * 2 * math.pi / 4.0)
        st.head_roll += breathe * 1.2
        st.head_pitch += math.sin(t * 2 * math.pi / 3.3 + 1) * 0.04

        # ---- auto blink ----
        for bt in blink_times:
            d = t - bt
            if 0 <= d <= 0.18:
                close = 1 - abs(d - 0.09) / 0.09     # 0..1..0 triangle
                f = max(0.0, 1 - close)
                st.eye_open_l *= f
                st.eye_open_r *= f

        # ---- actions ----
        if "nod" in plan.actions:
            st.head_pitch += math.sin(t * 2 * math.pi * 1.1) * 0.55 * _win(t, plan.duration)
        if "shake" in plan.actions:
            st.head_yaw += math.sin(t * 2 * math.pi * 1.3) * 0.7 * _win(t, plan.duration)
        if "look_left" in plan.actions:
            st.pupil_dx = -0.85 * _hold(t, plan.duration)
        if "look_right" in plan.actions:
            st.pupil_dx = 0.85 * _hold(t, plan.duration)
        if "look_up" in plan.actions:
            st.pupil_dy = -0.85 * _hold(t, plan.duration)
        if "look_down" in plan.actions:
            st.pupil_dy = 0.85 * _hold(t, plan.duration)
        if "yawn" in plan.actions:
            # one slow cycle: mouth opens wide, eyes squint, head tips back then down
            ph = min(1.0, t / max(0.1, plan.duration))
            wide = math.sin(ph * math.pi)            # 0..1..0
            st.mouth_open = max(st.mouth_open, wide * 0.95)
            sq = 1 - wide * 0.7
            st.eye_open_l *= sq
            st.eye_open_r *= sq
            st.head_pitch += (-0.25 + wide * 0.5)
            st.brow_raise_l += wide * 8
            st.brow_raise_r += wide * 8
        if "laugh" in plan.actions:
            st.mouth_smile = max(st.mouth_smile, 0.8)
            st.mouth_open = max(st.mouth_open, 0.25 + 0.3 * abs(math.sin(t * 2 * math.pi * 3.0)))
            st.head_pitch += math.sin(t * 2 * math.pi * 3.0) * 0.18
            st.eye_open_l *= 0.7
            st.eye_open_r *= 0.7
        if talk:
            k = int(t * 7) % len(talk_phon)
            env = talk_phon[k]
            # smooth between phonemes
            frac = (t * 7) - int(t * 7)
            k2 = (k + 1) % len(talk_phon)
            openness = talk_phon[k] * (1 - frac) + talk_phon[k2] * frac
            st.mouth_open = max(st.mouth_open, openness * 0.7)
            if st.mouth_smile < 0:
                st.mouth_smile *= 0.4

        frames.append(st)
    return frames


def _win(t: float, dur: float) -> float:
    """Fade a transient action in/out over the clip."""
    return _smooth(min(t, dur - t) / 0.4) if dur > 0.8 else 1.0


def _hold(t: float, dur: float) -> float:
    """Ease-in, hold, ease-out for a sustained gaze."""
    return _smooth(t / 0.4) * _smooth((dur - t) / 0.4)


def build_script_timeline(plans: list[Plan], fps: int = 25) -> list[AnimationState]:
    """Concatenate beats, easing each one from the previous beat's end pose."""
    frames: list[AnimationState] = []
    start: AnimationState | None = None
    for p in plans:
        seg = build_timeline(p, fps, start_state=start)
        frames.extend(seg)
        if seg:
            start = seg[-1]
    return frames


def animate(text: str, fps: int = 25) -> tuple[list[AnimationState], list[Plan]]:
    plans = make_script(text)
    return build_script_timeline(plans, fps), plans


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    for s in ["微笑", "非常生气并摇头", "惊讶地张大嘴", "说话: \"你好，我是莎士比亚\"",
              "向左看然后点头", "slightly sad"]:
        p = make_plan(s)
        print(f"{s!r:40} -> base={p.base:9} actions={p.actions} "
              f"intensity={p.intensity} dur={p.duration:.1f}")
