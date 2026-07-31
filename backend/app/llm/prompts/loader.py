"""Loads versioned prompt template files by (name, version), defaulting to
the latest version. Frontmatter (YAML header) is parsed but only lightly
used for now (phase/returns metadata); the body is the actual template
text with {placeholders} for str.format()-style substitution.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_VERSION = "v1"

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


@lru_cache
def _load_raw(name: str, version: str) -> tuple[dict, str]:
    path = _PROMPTS_DIR / version / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    frontmatter_text, body = match.groups()
    meta = {}
    for line in frontmatter_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body.strip()


def render(name: str, version: str = _DEFAULT_VERSION, **kwargs) -> str:
    _meta, body = _load_raw(name, version)
    if kwargs:
        return body.format(**kwargs)
    return body


def get_meta(name: str, version: str = _DEFAULT_VERSION) -> dict:
    meta, _body = _load_raw(name, version)
    return meta
