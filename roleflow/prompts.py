from __future__ import annotations

import re
from functools import cache

from .config import ROOT

SECTION = re.compile(r"^\[(system|user)\b[^\]]*\]\s*$", re.M)
SLOT = re.compile(r"\{(\w+)\}")


@cache
def load_prompt(name: str) -> tuple:
    path = ROOT / "prompts" / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt file {path}")
    text = path.read_text(encoding="utf-8").strip("\n")
    pieces = SECTION.split(text)
    if len(pieces) == 1:
        return (("user", text.strip()),)
    if pieces[0].strip():
        raise ValueError(f"{path}: text before the first [system] or [user] section")
    return tuple((pieces[i], pieces[i + 1].strip()) for i in range(1, len(pieces), 2))


def template(name: str) -> str:
    sections = load_prompt(name)
    if len(sections) != 1:
        raise ValueError(f"prompt {name} has {len(sections)} sections, not one")
    return sections[0][1]


def slots(text: str) -> set:
    return set(SLOT.findall(text))


def fill(text: str, /, **values) -> str:
    missing = slots(text) - set(values)
    if missing:
        raise KeyError(f"prompt slots not given: {sorted(missing)}")
    return SLOT.sub(lambda match: str(values[match.group(1)]), text)


def messages(name: str, /, **values) -> list:
    sections = load_prompt(name)
    unused = set(values) - set().union(*(slots(text) for _, text in sections))
    if unused:
        raise KeyError(f"prompt {name} has no slots {sorted(unused)}")
    return [{"role": role, "content": fill(text, **{k: v for k, v in values.items() if k in slots(text)})} for role, text in sections]
