"""Tests for gradetrail.spec: loading, validation, and run identity.

These encode the semantics in docs/design/eval-spec.md. If a test here needs
to change, the design doc changes first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gradetrail.errors import DatasetError, SpecError
from gradetrail.spec import compute_identity, load_spec

MINIMAL_YAML = """
name: gsm8k-subset
dataset:
  path: {dataset_path}
prompt: |
  Solve this. End with "Answer: <number>".

  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'Answer:\\s*{{{{ answer }}}}\\s*$'
"""


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "data.jsonl"
    rows = [
        {"id": "1", "question": "2+2?", "answer": "4"},
        {"id": "2", "question": "3+3?", "answer": "6"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


@pytest.fixture()
def spec_file(tmp_path: Path, dataset: Path) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(MINIMAL_YAML.format(dataset_path=dataset))
    return path


# --- loading ---------------------------------------------------------------


def test_minimal_spec_loads_with_defaults(spec_file: Path) -> None:
    spec = load_spec(spec_file)
    assert spec.name == "gsm8k-subset"
    assert spec.model.params.max_tokens == 1024
    assert spec.model.params.temperature == 0.0
    assert spec.run.concurrency == 8
    assert spec.run.max_retries == 5


def test_spec_is_immutable(spec_file: Path) -> None:
    spec = load_spec(spec_file)
    with pytest.raises(ValidationError):
        spec.name = "changed"  # type: ignore[misc]


def test_name_must_be_slug(spec_file: Path) -> None:
    text = spec_file.read_text().replace("gsm8k-subset", "Bad Name!")
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="name"):
        load_spec(spec_file)


def test_unknown_top_level_field_rejected(spec_file: Path) -> None:
    spec_file.write_text(spec_file.read_text() + "\nsurprise: true\n")
    with pytest.raises(SpecError, match="surprise"):
        load_spec(spec_file)


# --- model validation --------------------------------------------------------


def test_base_url_required_for_openai_compatible(spec_file: Path) -> None:
    text = spec_file.read_text().replace("provider: anthropic", "provider: openai_compatible")
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="base_url"):
        load_spec(spec_file)


def test_base_url_forbidden_for_anthropic(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "provider: anthropic", "provider: anthropic\n  base_url: http://localhost:8000"
    )
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="base_url"):
        load_spec(spec_file)


# --- scorer validation -------------------------------------------------------


def test_invalid_regex_fails_at_load(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "pattern: 'Answer:\\s*{{ answer }}\\s*$'", "pattern: '(unbalanced'"
    )
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="regex"):
        load_spec(spec_file)


def test_exact_scorer_parses(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'",
        "scorer:\n  type: exact\n  target_field: answer\n  normalize: [strip, lower]",
    )
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    assert spec.scorer.type == "exact"


# --- template strictness -----------------------------------------------------


def test_template_syntax_error_fails_at_load(spec_file: Path) -> None:
    text = spec_file.read_text().replace("{{ question }}", "{% broken")
    spec_file.write_text(text)
    with pytest.raises(SpecError, match="template"):
        load_spec(spec_file)


def test_missing_sample_field_fails_fast(spec_file: Path, dataset: Path) -> None:
    text = spec_file.read_text().replace("{{ question }}", "{{ nonexistent_field }}")
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    with pytest.raises(DatasetError, match="nonexistent_field"):
        spec.validate_against_dataset()


def test_valid_spec_passes_dataset_validation(spec_file: Path) -> None:
    spec = load_spec(spec_file)
    spec.validate_against_dataset()  # should not raise


def test_exact_scorer_missing_target_field_fails_fast(spec_file: Path, dataset: Path) -> None:
    text = spec_file.read_text().replace(
        "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'",
        "scorer:\n  type: exact\n  target_field: nonexistent_field",
    )
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    with pytest.raises(DatasetError, match="nonexistent_field"):
        spec.validate_against_dataset()


def test_exact_scorer_with_valid_target_field_passes_dataset_validation(spec_file: Path) -> None:
    text = spec_file.read_text().replace(
        "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'",
        "scorer:\n  type: exact\n  target_field: answer",
    )
    spec_file.write_text(text)
    spec = load_spec(spec_file)
    spec.validate_against_dataset()  # should not raise


# --- judge scorer: template strictness ----------------------------------------

VALID_JUDGE_YAML = """\
version: 1
output: score_0_1
prompt: |
  Question: {{ question }}
  Reference: {{ answer }}
  Response: {{ response }}

  Reply with only JSON: {"score": 0 or 1, "reason": "<one sentence>"}
"""

_JUDGE_SCORER_BLOCK = "scorer:\n  type: regex\n  pattern: 'Answer:\\s*{{ answer }}\\s*$'"


def _with_judge_scorer(spec_text: str, judge_prompt_path: str) -> str:
    return spec_text.replace(
        _JUDGE_SCORER_BLOCK,
        f"scorer:\n  type: judge\n  judge_prompt: {judge_prompt_path}\n  model:\n"
        "    provider: anthropic\n    name: claude-sonnet-4-6\n",
    )


@pytest.fixture()
def judge_file_path(tmp_path: Path) -> Path:
    path = tmp_path / "judge.yaml"
    path.write_text(VALID_JUDGE_YAML)
    return path


@pytest.fixture()
def judge_spec_file(tmp_path: Path, dataset: Path, judge_file_path: Path) -> Path:
    path = tmp_path / "eval.yaml"
    text = _with_judge_scorer(MINIMAL_YAML.format(dataset_path=dataset), str(judge_file_path))
    path.write_text(text)
    return path


def test_judge_prompt_response_placeholder_is_supplied_and_does_not_raise(
    tmp_path: Path, dataset: Path
) -> None:
    """{{ response }} is the one field validate_against_dataset() must inject itself:
    no real response exists yet at validation time, but every judge file references
    it (it's what's being graded). If the placeholder wiring is wrong, every judge
    scorer spec -- not just a buggy one -- would falsely fail validation."""
    judge_path = tmp_path / "judge.yaml"
    judge_path.write_text(
        "version: 1\noutput: score_0_1\nprompt: |\n  Q: {{ question }}\n  A: {{ response }}\n"
    )
    spec_path = tmp_path / "eval.yaml"
    spec_text = _with_judge_scorer(MINIMAL_YAML.format(dataset_path=dataset), str(judge_path))
    spec_path.write_text(spec_text)
    spec = load_spec(spec_path)
    spec.validate_against_dataset()  # should not raise


def test_judge_scorer_with_valid_judge_file_passes_dataset_validation(
    judge_spec_file: Path,
) -> None:
    spec = load_spec(judge_spec_file)
    spec.validate_against_dataset()  # should not raise


def test_judge_prompt_missing_sample_field_fails_fast(
    tmp_path: Path, dataset: Path, judge_file_path: Path
) -> None:
    judge_file_path.write_text(VALID_JUDGE_YAML.replace("{{ answer }}", "{{ nonexistent_field }}"))
    spec_path = tmp_path / "eval.yaml"
    spec_text = _with_judge_scorer(MINIMAL_YAML.format(dataset_path=dataset), str(judge_file_path))
    spec_path.write_text(spec_text)
    spec = load_spec(spec_path)
    with pytest.raises(DatasetError, match="nonexistent_field"):
        spec.validate_against_dataset()


def test_judge_prompt_file_missing_fails_with_resolved_path(tmp_path: Path, dataset: Path) -> None:
    missing_path = tmp_path / "no_such_judge.yaml"
    spec_path = tmp_path / "eval.yaml"
    spec_text = _with_judge_scorer(MINIMAL_YAML.format(dataset_path=dataset), str(missing_path))
    spec_path.write_text(spec_text)
    spec = load_spec(spec_path)
    with pytest.raises(DatasetError, match="does not exist") as exc_info:
        spec.validate_against_dataset()
    assert str(missing_path) in str(exc_info.value)


def test_judge_prompt_file_invalid_schema_fails_at_validation(
    tmp_path: Path, dataset: Path, judge_file_path: Path
) -> None:
    judge_file_path.write_text(VALID_JUDGE_YAML.replace("output: score_0_1", "output: percent"))
    spec_path = tmp_path / "eval.yaml"
    spec_text = _with_judge_scorer(MINIMAL_YAML.format(dataset_path=dataset), str(judge_file_path))
    spec_path.write_text(spec_text)
    spec = load_spec(spec_path)
    with pytest.raises(DatasetError) as exc_info:
        spec.validate_against_dataset()
    # the re-wrap must carry the pydantic reason forward, not degrade into a
    # generic "judge file invalid" -- both the field and why it's wrong.
    assert "output" in str(exc_info.value)
    assert "score_0_1" in str(exc_info.value) or "binary" in str(exc_info.value)


def test_judge_prompt_relative_path_resolves_against_spec_base_dir(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judges_dir = tmp_path / "judges"
    judges_dir.mkdir()
    (judges_dir / "correctness.yaml").write_text(VALID_JUDGE_YAML)

    spec_path = tmp_path / "eval.yaml"  # spec lives alongside judges/, base_dir == tmp_path
    spec_path.write_text(
        _with_judge_scorer(MINIMAL_YAML.format(dataset_path=dataset), "judges/correctness.yaml")
    )
    spec = load_spec(spec_path)

    # Give this test teeth: chdir somewhere that does NOT contain judges/, so a
    # CWD-based (rather than base_dir-based) resolution would raise "does not
    # exist" here instead of silently passing by accident.
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    spec.validate_against_dataset()  # should not raise, even though CWD != tmp_path


# --- dataset loading: sample ids ----------------------------------------------


def test_missing_id_field_falls_back_to_original_line_number_despite_shuffle(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "no_ids.jsonl"
    rows = [{"question": f"q{i}"} for i in range(5)]  # no "id" field at all
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows))

    spec_path = tmp_path / "eval.yaml"
    spec_path.write_text(
        f"""
name: no-ids
dataset:
  path: {dataset_path}
  shuffle_seed: 42
prompt: |
  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'x'
"""
    )
    spec = load_spec(spec_path)
    samples = spec.load_samples()

    # shuffle_seed=42 must actually reorder the samples relative to the file,
    # otherwise this test would pass trivially without exercising the bug.
    assert [s["question"] for s in samples] != [f"q{i}" for i in range(5)]

    # but each sample's fallback id must still reflect its ORIGINAL file line,
    # not its position after the shuffle.
    by_question = {s["question"]: s["id"] for s in samples}
    assert by_question == {f"q{i}": str(i + 1) for i in range(5)}


def test_present_id_field_is_never_overwritten(tmp_path: Path) -> None:
    dataset_path = tmp_path / "with_ids.jsonl"
    rows = [{"id": "custom-a", "question": "q0"}, {"id": "custom-b", "question": "q1"}]
    dataset_path.write_text("\n".join(json.dumps(r) for r in rows))

    spec_path = tmp_path / "eval.yaml"
    spec_path.write_text(
        f"""
name: with-ids
dataset:
  path: {dataset_path}
prompt: |
  {{{{ question }}}}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'x'
"""
    )
    spec = load_spec(spec_path)
    samples = spec.load_samples()
    assert [s["id"] for s in samples] == ["custom-a", "custom-b"]


# --- identity ----------------------------------------------------------------


def test_run_block_does_not_affect_identity(spec_file: Path) -> None:
    id_default = compute_identity(load_spec(spec_file))
    spec_file.write_text(spec_file.read_text() + "\nrun:\n  concurrency: 32\n")
    id_tuned = compute_identity(load_spec(spec_file))
    assert id_default == id_tuned


def test_prompt_change_changes_identity(spec_file: Path) -> None:
    id_before = compute_identity(load_spec(spec_file))
    spec_file.write_text(spec_file.read_text().replace("Solve this.", "Solve it."))
    id_after = compute_identity(load_spec(spec_file))
    assert id_before != id_after


def test_dataset_content_change_changes_identity(spec_file: Path, dataset: Path) -> None:
    id_before = compute_identity(load_spec(spec_file))
    dataset.write_text(dataset.read_text() + '\n{"id": "3", "question": "5+5?", "answer": "10"}')
    id_after = compute_identity(load_spec(spec_file))
    assert id_before != id_after


def test_identity_is_stable_hex_digest(spec_file: Path) -> None:
    a = compute_identity(load_spec(spec_file))
    b = compute_identity(load_spec(spec_file))
    assert a == b
    assert len(a) == 64
    int(a, 16)  # valid hex
