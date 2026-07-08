"""Tests for gradetrail.results: SampleResult/RunSummary, summarize(), and JSONL I/O.

Encodes design doc rule 5: every sample terminates in exactly one state
(scored, provider_error, judge_error); summarize() reports counts of each and
computes mean score over scored samples only. Cost math is Decimal-only.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path

import pytest

from gradetrail.errors import ResultsError
from gradetrail.results import PRICING, RunSummary, SampleResult, read_jsonl, summarize, write_jsonl
from gradetrail.spec import ModelSpec

PRICED_MODEL = ModelSpec(provider="anthropic", name="claude-sonnet-4-6")
UNKNOWN_MODEL = ModelSpec(provider="anthropic", name="claude-nonexistent-model-xyz")
JUDGE_MODEL = ModelSpec(provider="anthropic", name="claude-haiku-4-5")
UNKNOWN_JUDGE_MODEL = ModelSpec(provider="anthropic", name="claude-nonexistent-judge-xyz")
GPT_4O_MINI_MODEL = ModelSpec(provider="openai", name="gpt-4o-mini")


def make_scored(
    sample_id: str,
    score: float,
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cached: bool = False,
    served_model: str | None = "fake-model",
    judge_input_tokens: int | None = None,
    judge_output_tokens: int | None = None,
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        state="scored",
        score=score,
        response_text="some response",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=123.4,
        cached=cached,
        detail="matched 'x'",
        served_model=served_model,
        judge_input_tokens=judge_input_tokens,
        judge_output_tokens=judge_output_tokens,
    )


def make_error(
    sample_id: str, state: str, detail: str = "boom", *, cached: bool = False
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        state=state,
        score=None,
        response_text=None if state == "provider_error" else "some response",
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        cached=cached,
        detail=detail,
    )


# --- SampleResult / RunSummary shape -------------------------------------------


def test_sample_result_is_frozen_dataclass() -> None:
    result = make_scored("1", 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.score = 0.0  # type: ignore[misc]


def test_run_summary_is_frozen_dataclass() -> None:
    summary = summarize([make_scored("1", 1.0)], PRICED_MODEL, wall_time_s=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.n_samples = 99  # type: ignore[misc]


def test_served_model_defaults_to_none() -> None:
    result = make_error("1", "provider_error")
    assert result.served_model is None


def test_served_model_can_be_set_on_scored_samples() -> None:
    result = make_scored("1", 1.0, served_model="claude-sonnet-4-6-20260115")
    assert result.served_model == "claude-sonnet-4-6-20260115"


def test_run_summary_constructs_directly_with_expected_fields() -> None:
    summary = RunSummary(
        n_samples=2,
        n_scored=1,
        n_provider_error=1,
        n_judge_error=0,
        mean_score=1.0,
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=Decimal("0.01"),
        wall_time_s=5.0,
        cache_hits=0,
    )
    assert summary.n_samples == 2
    assert summary.total_cost_usd == Decimal("0.01")


# --- summarize: closed set of states --------------------------------------------


def test_summarize_unknown_state_raises_instead_of_miscounting() -> None:
    # A typo'd/future state must not be silently dropped from every counter --
    # n_samples would then stop equaling n_scored + n_provider_error + n_judge_error
    # and nobody would notice until the numbers looked wrong in a report.
    results = [make_scored("1", 1.0), make_error("2", "provider-error")]  # hyphen typo
    with pytest.raises(ResultsError, match="2"):
        summarize(results, PRICED_MODEL, wall_time_s=1.0)


def test_summarize_unknown_state_error_names_the_bad_state() -> None:
    results = [make_error("7", "totally-unknown")]
    with pytest.raises(ResultsError, match="totally-unknown"):
        summarize(results, PRICED_MODEL, wall_time_s=1.0)


# --- summarize: counts and states ----------------------------------------------


def test_summarize_counts_per_state() -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.0),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
        make_error("5", "judge_error"),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=10.0)
    assert summary.n_samples == 5
    assert summary.n_scored == 2
    assert summary.n_provider_error == 1
    assert summary.n_judge_error == 2


def test_summarize_wall_time_passthrough() -> None:
    summary = summarize([make_scored("1", 1.0)], PRICED_MODEL, wall_time_s=42.5)
    assert summary.wall_time_s == 42.5


def test_summarize_cache_hits_counts_cached_samples() -> None:
    results = [
        make_scored("1", 1.0, cached=True),
        make_scored("2", 1.0, cached=False),
        make_error("3", "provider_error", cached=True),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.cache_hits == 2


# --- summarize: mean score excludes errors -------------------------------------


def test_summarize_mean_score_over_scored_samples_only() -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.5),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
        make_error("5", "judge_error"),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    # mean of [1.0, 0.5] = 0.75, over n=2 -- NOT diluted to 0.3 by treating the
    # three error samples as zeros, and NOT inflated by excluding them from
    # the denominator while still counting them in the numerator.
    assert summary.mean_score == pytest.approx(0.75)
    assert summary.n_samples == 5
    assert summary.n_scored == 2


def test_summarize_mean_score_exactly_half_with_errors_present() -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.0),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
        make_error("5", "judge_error"),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.mean_score == 0.5
    assert summary.n_samples == 5
    assert summary.n_scored == 2


def test_summarize_mean_score_is_none_when_no_scored_samples() -> None:
    results = [make_error("1", "provider_error"), make_error("2", "judge_error")]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.mean_score is None


def test_summarize_mean_score_all_scored() -> None:
    results = [make_scored("1", 1.0), make_scored("2", 0.0), make_scored("3", 1.0)]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.mean_score == pytest.approx(2 / 3)


# --- summarize: tokens ----------------------------------------------------------


def test_summarize_total_tokens_sum_across_all_samples_including_cached() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=100, output_tokens=50, cached=False),
        make_scored("2", 1.0, input_tokens=200, output_tokens=75, cached=True),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 125


def test_summarize_error_samples_contribute_zero_tokens_without_crashing() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=100, output_tokens=50),
        make_error("2", "provider_error"),  # input_tokens/output_tokens are None
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 50


# --- summarize: cost (Decimal only) ----------------------------------------------


def test_summarize_cost_hand_computed() -> None:
    input_price, output_price = PRICING[("anthropic", "claude-sonnet-4-6")]
    assert input_price == Decimal("3.00")
    assert output_price == Decimal("15.00")

    results = [
        make_scored("1", 1.0, input_tokens=200_000, output_tokens=100_000),
        make_scored("2", 1.0, input_tokens=300_000, output_tokens=100_000),
    ]
    # total input = 500_000 -> 0.5M * $3.00  = $1.50
    # total output = 200_000 -> 0.2M * $15.00 = $3.00
    # total                                   = $4.50
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd == Decimal("4.50")
    assert isinstance(summary.total_cost_usd, Decimal)


def test_summarize_cost_is_zero_for_cached_samples_regardless_of_tokens() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=1_000_000, cached=True),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd == Decimal("0")
    # tokens are still counted in the totals even though they cost nothing
    assert summary.total_input_tokens == 1_000_000
    assert summary.total_output_tokens == 1_000_000


def test_summarize_mixed_cached_and_uncached_cost_only_counts_uncached() -> None:
    results = [
        make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=0, cached=True),
        make_scored("2", 1.0, input_tokens=1_000_000, output_tokens=0, cached=False),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd == Decimal("3.00")  # only sample 2's input tokens billed


def test_summarize_unknown_model_gives_none_cost_never_a_crash() -> None:
    results = [make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=1_000_000)]
    summary = summarize(results, UNKNOWN_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd is None


# --- summarize: gpt-4o-mini pricing (previously missing entirely) ----------------


def test_gpt_4o_mini_is_priced() -> None:
    # Root cause of the "Cost: unknown" bug on two real gpt-4o-mini runs
    # (951 in / 2775 out tokens, reported unknown): PRICING simply had no
    # ("openai", "gpt-4o-mini") entry at all -- confirmed by reading
    # summarize()'s lookup key, which is the requested spec.model.name, not
    # the API-reported served_model, so this was never a lookup-key mismatch.
    # $0.15/$0.60 per million tokens, unchanged since the 2024-07 launch.
    assert PRICING[("openai", "gpt-4o-mini")] == (Decimal("0.15"), Decimal("0.60"))


def test_summarize_partial_success_run_on_gpt_4o_mini_produces_real_cost() -> None:
    # The real bug scenario: a run with a mix of scored and provider_error
    # samples on a now-priced model must produce an actual dollar figure, not
    # "unknown" -- provider_error samples contribute zero tokens (None ->
    # 0), scored samples bill normally.
    results = [
        make_scored("1", 1.0, input_tokens=600_000, output_tokens=1_800_000),
        make_scored("2", 1.0, input_tokens=351_000, output_tokens=975_000),
        make_error("3", "provider_error"),
    ]
    summary = summarize(results, GPT_4O_MINI_MODEL, wall_time_s=1.0)
    # total input = 951_000 -> 0.951M * $0.15 = $0.14265
    # total output = 2_775_000 -> 2.775M * $0.60 = $1.665
    # total                                     = $1.80765
    assert summary.total_cost_usd == Decimal("1.80765")
    assert isinstance(summary.total_cost_usd, Decimal)
    assert summary.cost_unpriced_models == ()


def test_summarize_still_none_for_a_genuinely_unpriced_model() -> None:
    # Regression guard: adding gpt-4o-mini must not make the strict-None
    # behavior (Fix 2) disappear for models that are still genuinely absent
    # from PRICING.
    results = [make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=1_000_000)]
    summary = summarize(results, UNKNOWN_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd is None
    assert any("claude-nonexistent-model-xyz" in m for m in summary.cost_unpriced_models)


def test_summarize_empty_results_list() -> None:
    summary = summarize([], PRICED_MODEL, wall_time_s=0.0)
    assert summary.n_samples == 0
    assert summary.mean_score is None
    assert summary.total_cost_usd == Decimal("0")


# --- summarize: judge tokens and cost --------------------------------------------


def test_sample_result_judge_token_fields_default_to_none() -> None:
    result = make_scored("1", 1.0)
    assert result.judge_input_tokens is None
    assert result.judge_output_tokens is None


def test_run_summary_judge_fields_default_appropriately() -> None:
    summary = RunSummary(
        n_samples=1,
        n_scored=1,
        n_provider_error=0,
        n_judge_error=0,
        mean_score=1.0,
        total_input_tokens=10,
        total_output_tokens=5,
        total_cost_usd=Decimal("0.01"),
        wall_time_s=1.0,
        cache_hits=0,
    )
    assert summary.total_judge_input_tokens == 0
    assert summary.total_judge_output_tokens == 0
    assert summary.cost_unpriced_models == ()


def test_summarize_judge_tokens_totaled_separately_from_primary_tokens() -> None:
    results = [
        make_scored(
            "1",
            1.0,
            input_tokens=100,
            output_tokens=50,
            judge_input_tokens=200,
            judge_output_tokens=80,
        ),
        make_scored(
            "2",
            1.0,
            input_tokens=150,
            output_tokens=60,
            judge_input_tokens=300,
            judge_output_tokens=120,
        ),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0, judge_model=PRICED_MODEL)
    # primary and judge totals must not bleed into each other
    assert summary.total_input_tokens == 250
    assert summary.total_output_tokens == 110
    assert summary.total_judge_input_tokens == 500
    assert summary.total_judge_output_tokens == 200


def test_summarize_non_judge_eval_has_zero_judge_totals() -> None:
    results = [make_scored("1", 1.0)]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0)  # no judge_model
    assert summary.total_judge_input_tokens == 0
    assert summary.total_judge_output_tokens == 0


def test_summarize_judge_cost_added_to_primary_cost_when_both_priced() -> None:
    primary_input, primary_output = PRICING[("anthropic", "claude-sonnet-4-6")]
    judge_input, judge_output = PRICING[("anthropic", "claude-haiku-4-5")]
    assert (primary_input, primary_output) == (Decimal("3.00"), Decimal("15.00"))
    assert (judge_input, judge_output) == (Decimal("0.25"), Decimal("1.25"))

    results = [
        make_scored(
            "1",
            1.0,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            judge_input_tokens=1_000_000,
            judge_output_tokens=1_000_000,
        ),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0, judge_model=JUDGE_MODEL)
    # primary: 1M*$3.00 + 1M*$15.00 = $18.00; judge: 1M*$0.25 + 1M*$1.25 = $1.50
    assert summary.total_cost_usd == Decimal("19.50")


def test_summarize_judge_cost_billed_even_when_primary_response_is_cached() -> None:
    # Judge calls are never cached (see NOTES.md) -- a cache hit on the
    # primary response must not zero out judge cost the way it zeroes out
    # primary cost.
    results = [
        make_scored(
            "1",
            1.0,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cached=True,
            judge_input_tokens=1_000_000,
            judge_output_tokens=1_000_000,
        ),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0, judge_model=JUDGE_MODEL)
    assert summary.cache_hits == 1
    assert summary.total_cost_usd == Decimal("1.50")  # primary cost is $0 (cached); judge is not


def test_summarize_cost_none_when_primary_model_unpriced_names_it() -> None:
    results = [make_scored("1", 1.0, input_tokens=1_000_000, output_tokens=1_000_000)]
    summary = summarize(results, UNKNOWN_MODEL, wall_time_s=1.0)
    assert summary.total_cost_usd is None
    assert any(
        "primary" in m and "claude-nonexistent-model-xyz" in m for m in summary.cost_unpriced_models
    )


def test_summarize_cost_none_when_judge_model_unpriced_even_though_primary_is_priced() -> None:
    results = [
        make_scored(
            "1",
            1.0,
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            judge_input_tokens=1_000_000,
            judge_output_tokens=1_000_000,
        ),
    ]
    summary = summarize(results, PRICED_MODEL, wall_time_s=1.0, judge_model=UNKNOWN_JUDGE_MODEL)
    assert summary.total_cost_usd is None  # strict: an unpriced judge model taints the whole total
    assert any(
        "judge" in m and "claude-nonexistent-judge-xyz" in m for m in summary.cost_unpriced_models
    )


def test_summarize_cost_none_names_both_when_primary_and_judge_unpriced() -> None:
    results = [make_scored("1", 1.0)]
    summary = summarize(results, UNKNOWN_MODEL, wall_time_s=1.0, judge_model=UNKNOWN_JUDGE_MODEL)
    assert summary.total_cost_usd is None
    assert len(summary.cost_unpriced_models) == 2


# --- round trip: judge token fields ------------------------------------------------


def test_round_trip_preserves_judge_token_fields(tmp_path: Path) -> None:
    result = make_scored("1", 1.0, judge_input_tokens=2750, judge_output_tokens=800)
    path = tmp_path / "results.jsonl"
    write_jsonl([result], path)
    [round_tripped] = read_jsonl(path)
    assert round_tripped == result
    assert round_tripped.judge_input_tokens == 2750
    assert round_tripped.judge_output_tokens == 800


def test_round_trip_preserves_none_judge_fields_on_non_judge_result(tmp_path: Path) -> None:
    result = make_scored("1", 1.0)  # judge_input_tokens/judge_output_tokens default None
    assert result.judge_input_tokens is None
    path = tmp_path / "results.jsonl"
    write_jsonl([result], path)
    [round_tripped] = read_jsonl(path)
    assert round_tripped.judge_input_tokens is None
    assert round_tripped.judge_output_tokens is None


# --- write_jsonl / read_jsonl ----------------------------------------------------


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    results = [
        make_scored("1", 1.0),
        make_scored("2", 0.0, cached=True),
        make_error("3", "provider_error"),
        make_error("4", "judge_error"),
    ]
    path = tmp_path / "results.jsonl"
    write_jsonl(results, path)
    round_tripped = read_jsonl(path)
    assert round_tripped == results


def test_round_trip_preserves_none_fields(tmp_path: Path) -> None:
    result = make_error("1", "provider_error")
    assert result.score is None
    assert result.response_text is None
    assert result.input_tokens is None

    path = tmp_path / "results.jsonl"
    write_jsonl([result], path)
    [round_tripped] = read_jsonl(path)
    assert round_tripped.score is None
    assert round_tripped.response_text is None
    assert round_tripped.input_tokens is None
    assert round_tripped.output_tokens is None
    assert round_tripped.latency_ms is None


def test_write_jsonl_produces_one_json_object_per_line(tmp_path: Path) -> None:
    results = [make_scored("1", 1.0), make_scored("2", 0.5), make_scored("3", 0.0)]
    path = tmp_path / "results.jsonl"
    write_jsonl(results, path)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)  # each line parses independently


def test_write_jsonl_field_order_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_jsonl([make_scored("1", 1.0)], path)
    [line] = path.read_text().splitlines()
    keys = list(json.loads(line).keys())
    expected = [f.name for f in dataclasses.fields(SampleResult)]
    assert keys == expected


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    result = make_scored("1", 1.0)
    path.write_text(json.dumps(dataclasses.asdict(result)) + "\n\n")
    assert read_jsonl(path) == [result]
