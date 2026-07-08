"""Tests for gradetrail.scorers.judge: judge-file loading and the LLM-as-judge scorer.

Mocks only the Provider boundary (a fake .complete()), with hand-built raw
reply strings covering malformed JSON, JSON wrapped in markdown fences, and
valid replies. Never calls a real API. Per design doc rule 5, a judge that
never produces parseable JSON (even after the one nudge retry) must return
state="judge_error" — never a silent score of 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gradetrail.errors import JudgeError, ProviderError
from gradetrail.providers.base import ProviderResponse
from gradetrail.scorers.judge import load_judge_file, score_judge
from gradetrail.spec import JudgeScorer, ModelSpec

VALID_JUDGE_YAML = """
version: 2
output: score_0_1
prompt: |
  Question: {{ question }}
  Expected answer: {{ answer }}
  Model response: {{ response }}

  Reply with only a JSON object: {"score": <0 or 1>, "reason": "<one sentence>"}
"""

SAMPLE = {"question": "2+2?", "answer": "4"}


@pytest.fixture()
def judge_path(tmp_path: Path) -> Path:
    path = tmp_path / "judge.yaml"
    path.write_text(VALID_JUDGE_YAML)
    return path


def make_scorer(*, samples: int = 1) -> JudgeScorer:
    return JudgeScorer(
        type="judge",
        judge_prompt="judge.yaml",
        model=ModelSpec(provider="anthropic", name="claude-sonnet-4-6"),
        samples=samples,
    )


class FakeProvider:
    """Replays a scripted sequence of outcomes for .complete(); records prompts sent."""

    def __init__(self, outcomes: list[str | Exception]) -> None:
        self._outcomes = outcomes
        self.prompts: list[str] = []

    async def complete(self, prompt: str, params: object) -> ProviderResponse:
        self.prompts.append(prompt)
        outcome = self._outcomes[len(self.prompts) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return ProviderResponse(
            text=outcome, input_tokens=10, output_tokens=5, latency_ms=1.0, model="judge-model"
        )


# --- load_judge_file -----------------------------------------------------------


def test_load_judge_file_parses_valid_file(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    assert judge_file.version == 2
    assert judge_file.output == "score_0_1"
    assert "{{ question }}" in judge_file.prompt


def test_load_judge_file_missing_file_raises_judge_error(tmp_path: Path) -> None:
    with pytest.raises(JudgeError, match="does not exist"):
        load_judge_file(tmp_path / "missing.yaml")


def test_load_judge_file_invalid_yaml_raises_judge_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("version: [2\n")
    with pytest.raises(JudgeError):
        load_judge_file(path)


def test_load_judge_file_rejects_unknown_output_type(tmp_path: Path) -> None:
    path = tmp_path / "judge.yaml"
    path.write_text(VALID_JUDGE_YAML.replace("output: score_0_1", "output: percent"))
    with pytest.raises(JudgeError, match="output"):
        load_judge_file(path)


def test_load_judge_file_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "judge.yaml"
    path.write_text(VALID_JUDGE_YAML + "\nsurprise: true\n")
    with pytest.raises(JudgeError, match="surprise"):
        load_judge_file(path)


# --- score_judge: happy path -----------------------------------------------------


async def test_score_judge_valid_reply_returns_scored_state(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['{"score": 1, "reason": "correct"}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "scored"
    assert result.score == 1.0
    assert result.detail == "correct"
    assert len(provider.prompts) == 1


async def test_score_judge_renders_sample_fields_and_response_into_prompt(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['{"score": 1, "reason": "correct"}'])
    await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    [prompt] = provider.prompts
    assert "2+2?" in prompt
    assert "Expected answer: 4" in prompt
    assert "Model response: 4" in prompt


async def test_score_judge_parses_json_wrapped_in_markdown_fence(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['```json\n{"score": 1, "reason": "fenced"}\n```'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "scored"
    assert result.score == 1.0
    assert result.detail == "fenced"


async def test_score_judge_parses_fence_without_json_language_tag(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['```\n{"score": 0, "reason": "plain fence"}\n```'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "scored"
    assert result.score == 0.0


# --- score_judge: malformed JSON & retry-with-nudge --------------------------------


async def test_score_judge_malformed_json_retries_once_with_nudge_then_succeeds(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(["not json at all", '{"score": 1, "reason": "recovered"}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "scored"
    assert result.score == 1.0
    assert len(provider.prompts) == 2
    assert provider.prompts[1].startswith(provider.prompts[0])
    assert len(provider.prompts[1]) > len(provider.prompts[0])  # nudge text was appended


async def test_score_judge_malformed_json_after_retry_returns_judge_error(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(["not json", "still not json"])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert result.detail  # some explanation, never blank
    assert len(provider.prompts) == 2  # exactly one retry, not more


async def test_score_judge_never_returns_silent_zero_on_parse_failure(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(["garbage", "still garbage"])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert not (result.state == "scored" and result.score == 0.0)
    assert result.state == "judge_error"


async def test_score_judge_missing_reason_field_is_judge_error(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['{"score": 1}', '{"score": 1}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert len(provider.prompts) == 2


async def test_score_judge_binary_output_rejects_non_binary_score(tmp_path: Path) -> None:
    path = tmp_path / "judge.yaml"
    path.write_text(VALID_JUDGE_YAML.replace("output: score_0_1", "output: binary"))
    judge_file = load_judge_file(path)
    provider = FakeProvider(['{"score": 0.5, "reason": "partial"}', '{"score": 1, "reason": "ok"}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "scored"  # recovered on the nudge retry
    assert result.score == 1.0
    assert len(provider.prompts) == 2


async def test_score_judge_score_0_1_rejects_out_of_range_score(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['{"score": 1.5, "reason": "bad"}', '{"score": 1.5, "reason": "bad"}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"


# --- score_judge: samples > 1 -------------------------------------------------------


async def test_score_judge_samples_greater_than_one_averages_scores_and_joins_reasons(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(
        [
            '{"score": 1, "reason": "first"}',
            '{"score": 0, "reason": "second"}',
            '{"score": 1, "reason": "third"}',
        ]
    )
    result = await score_judge(SAMPLE, "4", make_scorer(samples=3), judge_file, provider)
    assert result.state == "scored"
    assert result.score == pytest.approx(2 / 3)
    assert "first" in result.detail
    assert "second" in result.detail
    assert "third" in result.detail
    assert len(provider.prompts) == 3


async def test_score_judge_samples_greater_than_one_short_circuits_on_first_judge_error(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(
        [
            '{"score": 1, "reason": "first"}',
            "garbage",
            "still garbage",
        ]
    )
    result = await score_judge(SAMPLE, "4", make_scorer(samples=3), judge_file, provider)
    assert result.state == "judge_error"
    assert len(provider.prompts) == 3  # 1 good sample, then 1 bad + 1 nudge retry, then stop


# --- score_judge: provider failure --------------------------------------------------


async def test_score_judge_provider_error_surfaces_as_judge_error(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider([ProviderError("rate limit retries exhausted")])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert "rate limit" in result.detail
    assert len(provider.prompts) == 1  # no nudge retry for provider-level failures


async def test_score_judge_provider_error_detail_includes_chained_cause(
    judge_path: Path,
) -> None:
    """detail is the only debugging surface for a judge_error; it must carry the
    underlying cause, not just the outer 'retries exhausted' message."""
    judge_file = load_judge_file(judge_path)
    cause = RuntimeError("socket timeout after 30s")
    error = ProviderError("judge-model: exhausted 5 retries")
    error.__cause__ = cause
    provider = FakeProvider([error])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert "exhausted 5 retries" in result.detail
    assert "socket timeout after 30s" in result.detail


async def test_score_judge_parse_failure_detail_includes_truncated_raw_reply(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    raw_reply = "not valid json " * 20 + "TAIL_MARKER_SHOULD_BE_TRUNCATED_AWAY"
    assert len(raw_reply) > 200
    provider = FakeProvider([raw_reply, raw_reply])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert raw_reply[:50] in result.detail  # a meaningful prefix survives
    assert "TAIL_MARKER_SHOULD_BE_TRUNCATED_AWAY" not in result.detail  # tail is truncated away


# --- score_judge: token accounting ----------------------------------------------
#
# FakeProvider.complete() reports a fixed 10 input / 5 output tokens per call
# (see its definition above), so every assertion below is just 10*n_calls /
# 5*n_calls for however many real provider calls that scenario made.


async def test_score_judge_reports_tokens_consumed_even_when_it_ends_in_judge_error(
    judge_path: Path,
) -> None:
    """A judge that never produces parseable JSON still burned two real API
    calls (the original attempt and its one nudge retry) before giving up --
    those tokens cost money and must be reported, not dropped along with the
    score. Undercounting them here is the exact bug class this fix exists
    for: a judge_error run must not silently look free."""
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(["garbage", "still garbage"])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert result.judge_input_tokens == 20
    assert result.judge_output_tokens == 10


async def test_score_judge_reports_judge_tokens_on_single_successful_call(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(['{"score": 1, "reason": "correct"}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.judge_input_tokens == 10
    assert result.judge_output_tokens == 5


async def test_score_judge_sums_tokens_across_nudge_retry(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(["not json at all", '{"score": 1, "reason": "recovered"}'])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "scored"
    # both the malformed first call and the nudge retry cost real tokens
    assert result.judge_input_tokens == 20
    assert result.judge_output_tokens == 10


async def test_score_judge_sums_tokens_across_multiple_samples(judge_path: Path) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(
        [
            '{"score": 1, "reason": "first"}',
            '{"score": 0, "reason": "second"}',
            '{"score": 1, "reason": "third"}',
        ]
    )
    result = await score_judge(SAMPLE, "4", make_scorer(samples=3), judge_file, provider)
    assert result.judge_input_tokens == 30
    assert result.judge_output_tokens == 15


async def test_score_judge_reports_zero_tokens_when_provider_never_returns_a_response(
    judge_path: Path,
) -> None:
    # A ProviderError means .complete() itself raised -- no ProviderResponse
    # was ever obtained, so there is genuinely nothing to bill. 0, not None:
    # this is still a judge scorer, the token count is just legitimately zero.
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider([ProviderError("rate limit retries exhausted")])
    result = await score_judge(SAMPLE, "4", make_scorer(), judge_file, provider)
    assert result.state == "judge_error"
    assert result.judge_input_tokens == 0
    assert result.judge_output_tokens == 0


async def test_score_judge_samples_greater_than_one_short_circuit_reports_partial_tokens(
    judge_path: Path,
) -> None:
    judge_file = load_judge_file(judge_path)
    provider = FakeProvider(
        [
            '{"score": 1, "reason": "first"}',  # 1 good call: 10/5
            "garbage",  # bad call: 10/5
            "still garbage",  # nudge retry: 10/5
        ]
    )
    result = await score_judge(SAMPLE, "4", make_scorer(samples=3), judge_file, provider)
    assert result.state == "judge_error"
    assert result.judge_input_tokens == 30
    assert result.judge_output_tokens == 15
