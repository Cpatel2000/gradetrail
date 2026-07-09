"""Build examples/data/triviaqa_100.jsonl from TriviaQA's validation split.

Requires the `datasets` library (pip install datasets) -- script-only, NOT a
gradetrail dependency, deliberately not added to pyproject.toml. This script is
run once to regenerate the file; gradetrail itself never imports `datasets`.

Takes the first 100 samples of the `rc.nocontext` config's validation split
(streaming, so the full TriviaQA download is never pulled), each row:
{"id", "question", "answer", "aliases"} where `answer` is TriviaQA's primary
answer (`answer.value`) and `aliases` is the full accepted-answers list
(`answer.aliases`). The four-field schema is deliberate: the row carries the
raw answers only, no precomputed regex fragment (unlike gsm8k_500.jsonl's
`answer_pattern`) -- this dataset feeds a scorer-comparison experiment
(examples/triviaqa_100_exact.yaml vs examples/triviaqa_100_judge.yaml) that
quantifies how much a primary-answer-only regex understates accuracy versus
an alias-aware judge on the same responses.

Because the exact spec interpolates the raw `answer` into its regex pattern
(there is no regex-escape jinja filter in the spec environment), this script
verifies on every invocation that each selected answer renders into the exact
spec's pattern template, compiles, and matches the answer itself. A sample
whose answer breaks that property fails the build loudly rather than failing
sample N of a paid run.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from datasets import load_dataset

_N_SAMPLES = 100
_OUT_PATH = Path(__file__).parent.parent / "examples" / "data" / "triviaqa_100.jsonl"

# Must stay in sync with examples/triviaqa_100_exact.yaml's scorer.pattern
# (which additionally sets flags: [IGNORECASE]).
_SPEC_PATTERN_TEMPLATE = r"\b{answer}\b"


def _verify_row(row: dict) -> None:
    """The exact spec's pattern must compile and self-match for this answer."""
    pattern = _SPEC_PATTERN_TEMPLATE.format(answer=row["answer"])
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise SystemExit(
            f"sample {row['id']}: answer {row['answer']!r} does not compile as a "
            f"regex fragment ({e}); the exact spec would fail mid-run on this "
            "sample -- pick a different slice or add escaping to the spec"
        ) from None
    if not compiled.search(row["answer"]):
        raise SystemExit(
            f"sample {row['id']}: pattern {pattern!r} does not match its own "
            f"answer {row['answer']!r}"
        )


def main() -> None:
    stream = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True
    )
    rows: list[dict] = []
    for item in itertools.islice(stream, _N_SAMPLES):
        row = {
            "id": item["question_id"],
            "question": item["question"],
            "answer": item["answer"]["value"],
            "aliases": item["answer"]["aliases"],
        }
        _verify_row(row)
        rows.append(row)
    assert len(rows) == _N_SAMPLES, f"expected {_N_SAMPLES} samples, got {len(rows)}"

    _OUT_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))

    multi = sum(1 for r in rows if len(r["aliases"]) > 1)
    max_aliases = max(len(r["aliases"]) for r in rows)
    total = sum(len(r["aliases"]) for r in rows)
    print(f"wrote {len(rows)} samples to {_OUT_PATH}")
    print(
        f"alias stats: {multi}/{len(rows)} samples have >1 accepted alias; "
        f"max {max_aliases}; mean {total / len(rows):.1f}"
    )


if __name__ == "__main__":
    main()
