"""Build examples/data/gsm8k_500.jsonl from the GSM8K test split.

Requires the `datasets` library (pip install datasets) -- script-only, NOT an
evalflow dependency, deliberately not added to pyproject.toml. This script is
run once to regenerate the file; evalflow itself never imports `datasets`.

Each row gets an `answer_pattern` field alongside `answer`: the same number
with optional commas at thousands boundaries (e.g. "114200" -> "114,?200",
matching either "114200" or "114,200"). 52/500 answers in this slice are >=4
digits, where a model could plausibly write either form -- see NOTES.md and
examples/gsm8k_500.yaml, which uses answer_pattern (not answer) in its regex.
"""

from __future__ import annotations

import json

from datasets import load_dataset


def _comma_tolerant_pattern(answer: str) -> str:
    """`answer` (a plain integer string, optionally negative) as a regex
    fragment that also accepts thousands-comma formatting: "2125" ->
    "2,?125", matching both "2125" and "2,125"."""
    sign = ""
    digits = answer
    if digits.startswith("-"):
        sign, digits = "-", digits[1:]
    grouped: list[str] = []
    for i, ch in enumerate(reversed(digits)):
        if i and i % 3 == 0:
            grouped.append(",?")
        grouped.append(ch)
    return sign + "".join(reversed(grouped))


def main() -> None:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    with open("examples/data/gsm8k_500.jsonl", "w") as f:
        for i, row in enumerate(ds.select(range(500))):
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            sample = {
                "id": str(i + 1),
                "question": row["question"],
                "answer": answer,
                "answer_pattern": _comma_tolerant_pattern(answer),
            }
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
