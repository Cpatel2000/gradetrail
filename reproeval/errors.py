"""Exception hierarchy for reproeval.

All reproeval code raises these instead of bare exceptions so callers can
distinguish spec problems from runtime problems.
"""

from __future__ import annotations


class ReproevalError(Exception):
    """Base class for all reproeval errors."""


class SpecError(ReproevalError):
    """The eval spec is invalid. Message names the field and the fix."""


class DatasetError(ReproevalError):
    """The dataset file is missing, malformed, or incompatible with the spec."""


class ProviderError(ReproevalError):
    """A model provider call failed after retries were exhausted."""


class JudgeError(ReproevalError):
    """A judge response could not be parsed or the judge file is invalid."""


class CacheError(ReproevalError):
    """The response cache was misused (e.g. accessed before connect())."""


class ResultsError(ReproevalError):
    """A SampleResult had a state outside the closed set summarize() knows how to count."""
