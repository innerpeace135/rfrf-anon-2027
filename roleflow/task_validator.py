from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Deficit:
    category: str
    trace: str
    suggested_capability: str

    def to_dict(self) -> dict:
        return {"category": self.category, "trace": self.trace, "suggested_capability": self.suggested_capability}


@dataclass(frozen=True)
class Validation:
    verdict: int
    score: float
    deficit: Optional[Deficit] = None
    error_output: str = ""

    def summary(self) -> str:
        return f"{self.deficit.category}: {self.deficit.trace}" if self.deficit else "none"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "score": self.score, "deficit": self.deficit.to_dict() if self.deficit else None}


class TaskValidator:
    def assess(self, output: str, trace: list) -> Validation:
        raise NotImplementedError
