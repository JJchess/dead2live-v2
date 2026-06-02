"""Lightweight config + secret loading (no external deps).

Reads a project-root ``.env`` (KEY=VALUE per line) into os.environ if present,
without overriding variables already set in the real environment.
"""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]   # project root (dead2live/)


def load_dotenv(path: str | Path | None = None) -> None:
    p = Path(path) if path else _ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


load_dotenv()


def get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


GEMINI_API_KEY = get("GEMINI_API_KEY")
GEMINI_MODEL = get("GEMINI_MODEL", "gemini-2.0-flash")
