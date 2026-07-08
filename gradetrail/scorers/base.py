"""Shared scoring result type for all scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring one sample's response.

    score is meaningful only when state == "scored"; error states carry a
    placeholder 0.0 and put the failure detail in `detail` instead — the only
    debugging surface available once a run summary reduces to error counts.
    """

    score: float
    state: Literal["scored", "judge_error"]
    detail: str
    # Only ever set by the judge scorer (None from score_exact/score_regex).
    # Judge calls are never cached, so these are real tokens spent even when
    # the primary response was a cache hit -- see NOTES.md.
    judge_input_tokens: int | None = None
    judge_output_tokens: int | None = None
