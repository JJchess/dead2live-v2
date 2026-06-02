"""Gemini-powered instruction understanding.

Turns a free-form, conversational instruction (any language) into a structured
multi-beat animation *script* using our fixed vocabulary of expressions and
actions. Falls back silently to the rule-based parser on any error, so the app
always works offline too.

Enable with::

    from dead2live import brain_gemini
    brain_gemini.enable()        # sets brain.llm_parse_script
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error

from . import config, brain
from .brain import Plan, EXPRESSIONS, _ACTION_KEYWORDS

_EXPRS = list(EXPRESSIONS.keys())
_ACTS = list(_ACTION_KEYWORDS.keys())

_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent?key={key}")

_SYSTEM = f"""You are the motion director for a 2D digital-human puppet.
Convert the user's instruction (any language) into a JSON animation script:
a list of sequential "beats". Each beat has:
- base: ONE facial expression, from {_EXPRS}
- intensity: 0.4 (subtle) .. 1.6 (strong), default 1.0
- actions: subset of {_ACTS} (may be empty). 'talk' = moving mouth speech.
- duration: seconds for this beat, 1.0 .. 8.0
- speak_text: the words to "speak" if the user asks the character to say
  something, else null. Used to time the mouth.

Rules:
- Split compound or sequential instructions ("first smile, then look surprised
  and shake head") into multiple beats in order.
- Map feelings/situations to the closest expression. Mapping hints:
  * 无语 / 无奈 / 翻白眼 / 醉了 / 服了 / 一言难尽 / "whatever" -> base "speechless"
    (optionally + 'shake' for a small head shake of disbelief). Do NOT use
    "surprised" for 无语.
  * 嫌弃 / 恶心 / 鄙视 / disgusted -> base "disgust"
  * 累 / 困 / tired / sleepy -> base "sad" or "neutral" + 'yawn'
  * 困惑 / confused / hmm -> base "thinking"
  * 开心大笑 / 哈哈 -> base "happy" + 'laugh'
  * 打招呼 / greet / say something -> base "neutral"/"happy" + 'talk' with speak_text
- Prefer "thinking" or "neutral" over "surprised" when the feeling is subtle.
- A single simple instruction = one beat.
- Keep it short: 1-4 beats. Output ONLY JSON matching the schema."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "enum": _EXPRS},
                    "intensity": {"type": "number"},
                    "actions": {"type": "array",
                                "items": {"type": "string", "enum": _ACTS}},
                    "duration": {"type": "number"},
                    "speak_text": {"type": "string"},
                },
                "required": ["base", "intensity", "actions", "duration"],
            },
        }
    },
    "required": ["beats"],
}


def _call(text: str) -> dict:
    key = config.GEMINI_API_KEY
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = _ENDPOINT.format(model=config.GEMINI_MODEL, key=key)
    body = {
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0.4,
            "response_mime_type": "application/json",
            "response_schema": _SCHEMA,
        },
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    parts = resp["candidates"][0]["content"]["parts"]
    raw = "".join(p.get("text", "") for p in parts)
    return json.loads(raw)


def parse_script(text: str) -> list[Plan]:
    data = _call(text)
    beats = data.get("beats", [])
    plans: list[Plan] = []
    for i, b in enumerate(beats):
        base = b.get("base", "neutral")
        if base not in EXPRESSIONS:
            base = "neutral"
        actions = [a for a in b.get("actions", []) if a in _ACTION_KEYWORDS]
        speak = b.get("speak_text") or None
        if isinstance(speak, str) and speak.strip().lower() in ("null", "none", ""):
            speak = None
        if speak and "talk" not in actions:
            actions.append("talk")
        dur = float(b.get("duration", 3.0))
        dur = max(1.0, min(8.0, dur))
        plans.append(Plan(
            base=base,
            intensity=float(b.get("intensity", 1.0)),
            actions=actions,
            duration=dur,
            speak_text=speak,
            seed=(abs(hash(text)) + i * 911) % 100000,
        ))
    if not plans:
        raise ValueError("empty beats")
    return plans


def enable() -> bool:
    """Wire Gemini into the brain. Returns True if a key is available."""
    if not config.GEMINI_API_KEY:
        return False
    brain.llm_parse_script = parse_script
    return True


if __name__ == "__main__":
    enable()
    tests = [
        "你看起来有点累，打个哈欠然后摇摇头",
        "先开心地笑一下，然后突然很惊讶地张大嘴巴",
        '跟我打个招呼，说"大家好，我是莎士比亚"，然后点点头',
        "我讲了个冷笑话，你做出无语又有点生气的表情",
    ]
    for t in tests:
        try:
            plans = parse_script(t)
            print(f"\n{t}")
            for p in plans:
                print(f"   beat base={p.base:9} act={p.actions} "
                      f"int={p.intensity} dur={p.duration:.1f} speak={p.speak_text!r}")
        except Exception as e:
            print(t, "-> ERR", type(e).__name__, str(e)[:200])
