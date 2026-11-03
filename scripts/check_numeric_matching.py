"""
Checks the numeric model's question matching on CLOSED questions.

For every closed or resolved numeric/discrete question in the given
tournaments, this asks the helper model to match it to a known series, runs
the checks in numeric_model/matching.py, and prints what was decided. For
accepted matches it also shows the model's forecast made with data up to the
day the question opened (so no later data leaks in) next to the actual
resolution.

It submits nothing and never looks at open questions: tuning a bot on open
questions in a scored tournament breaks Metaculus's "no human in the loop"
rule.

Usage (needs METACULUS_TOKEN and an AI key, as for the bot):
    python scripts/check_numeric_matching.py [--tournaments fall-futureeval-2026,minibench] [--limit 200]

Exits non-zero if a question that should be rejected was accepted, because a
wrong match is the costly mistake.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting_tools import MetaculusClient, NumericQuestion  # noqa: E402
from forecasting_tools.helpers.metaculus_client import ApiFilter  # noqa: E402

from main import pick_models, question_info  # noqa: E402
from numeric_model.matching import MatchResult, match_question  # noqa: E402

# Closed questions from fall 2026 whose right answer is known, keyed by words
# in the title. These must all be rejected: the model has no data for them, or
# its data measures something slightly different.
MUST_REJECT = {
    "intraday": "the model only has daily closes, not intraday highs",
    "lower atlantic": "FRED's GASDESW is national diesel, not the lower-Atlantic region",
    "nvda": "single stocks aren't in the model yet",
    "nvidia": "single stocks aren't in the model yet",
    "payroll": "payrolls aren't in the model (and revisions aren't handled)",
}


def expected_rejection(title: str) -> str | None:
    lowered = title.lower()
    for word, why in MUST_REJECT.items():
        if word in lowered:
            return why
    return None


async def check(question: NumericQuestion, helper) -> tuple[MatchResult, str]:
    opened = question.open_time or question.published_time or datetime.now(timezone.utc)
    result = await match_question(question_info(question), helper.invoke, as_of=opened.date())
    outcome = ""
    resolution = question.resolution_string
    if result.accepted:
        low, mid, high = (result.scaled_quantile(q) for q in (0.1, 0.5, 0.9))
        outcome = f"model 10/50/90%: {low:,.4g} / {mid:,.4g} / {high:,.4g}; resolved {resolution}"
        try:
            value = float(resolution) if resolution is not None else None
        except ValueError:
            value = None
        if value is not None:
            outcome += " (inside 80% range)" if low <= value <= high else " (OUTSIDE 80% range)"
    return result, outcome


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournaments", default="")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    client = MetaculusClient()
    tournaments: list[str | int] = [
        int(t) if t.isdigit() else t for t in args.tournaments.split(",") if t.strip()
    ] or [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]
    questions = await client.get_questions_matching_filter(
        ApiFilter(
            allowed_types=["numeric", "discrete"],
            allowed_statuses=["closed", "resolved"],
            allowed_tournaments=tournaments,
        ),
        num_questions=args.limit,
        error_if_question_target_missed=False,
    )
    # Belt and braces: never touch a question that is still open.
    now = datetime.now(timezone.utc)
    questions = [
        q for q in questions
        if isinstance(q, NumericQuestion) and q.close_time is not None and q.close_time <= now
    ]
    print(f"Checking {len(questions)} closed numeric questions from {tournaments}\n")

    helper = pick_models()["parser"]
    lines: list[str] = []
    wrong_accepts = 0
    accepted = 0
    for question in questions:
        try:
            result, outcome = await check(question, helper)
        except Exception as error:
            lines.append(f"ERROR  {question.question_text}\n       {error}")
            continue
        why_reject = expected_rejection(question.question_text)
        flag = ""
        if result.accepted:
            accepted += 1
            if why_reject:
                wrong_accepts += 1
                flag = f"  <-- WRONG: should be rejected ({why_reject})"
        status = "ACCEPT" if result.accepted else "reject"
        lines.append(
            f"{status} {question.question_text}{flag}\n"
            f"       {question.page_url}\n"
            f"       {'; '.join(result.reasons)}"
            + (f"\n       {outcome}" if outcome else "")
        )

    report = "\n".join(lines)
    summary = (
        f"\n{accepted} of {len(questions)} accepted; "
        f"{wrong_accepts} accepted that should have been rejected."
    )
    print(report + summary)
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write("## Numeric model matching on closed questions\n\n```\n")
            handle.write(report + summary + "\n```\n")
    return 1 if wrong_accepts else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
