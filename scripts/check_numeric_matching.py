"""
Checks the numeric model's question matching on CLOSED questions.

For every closed or resolved numeric/discrete question in the given
tournaments (group questions are split into their subquestions, as the live
bot does), this asks the helper model to match it to a known series, runs the
checks in numeric_model/matching.py, and prints what was decided.

Metaculus withholds the description, resolution criteria, fine print and
resolution of closed questions an account didn't forecast (its API gateway
rules, documented in Metaculus PR #5261). So matching here works on titles,
units and ranges alone, with the live check for missing criteria turned off.
That makes it a harder test than live use, where open questions come with
their full text.

For accepted matches it also shows:
- the text the forecasting prompt would get,
- the model's forecast made with data up to the day before the question
  opened (so no later data leaks in) next to what actually happened,
  read from the series itself (for an exact-source match, that is how the
  question resolved),
- a dry run of the bot's real blending code on the real question, with a
  stand-in AI forecast, to check that the blend is accepted.

It also reports how often a group subquestion's title leaves out its label
(e.g. which index it is about), which matters for every prompt the bot writes.

It submits nothing and never looks at open questions: tuning a bot on open
questions in a scored tournament breaks Metaculus's "no human in the loop"
rule.

Usage (needs METACULUS_TOKEN and an AI key, as for the bot):
    python scripts/check_numeric_matching.py [--tournaments market-pulse-26q3,33022] [--limit 300]

Exits non-zero if a question that should be rejected was accepted (a wrong
match is the costly mistake) or if a blend failed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting_tools import (  # noqa: E402
    MetaculusClient,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    ReasonedPrediction,
)
from forecasting_tools.helpers.metaculus_client import ApiFilter  # noqa: E402

from main import FallTemplateBot2026, pick_models, question_info  # noqa: E402
from numeric_model.data_sources import fetch_fred  # noqa: E402
from numeric_model.matching import MatchResult, anchor_text, match_question  # noqa: E402

# Questions whose right answer is known, keyed by words in the title. These
# must all be rejected: the model has no data for them, or its data measures
# something slightly different.
MUST_REJECT = {
    "intraday": "the model only has daily closes, not intraday highs",
    "lower atlantic": "FRED's GASDESW is national diesel, not the lower-Atlantic region",
    "nvda": "single stocks aren't in the model yet",
    "nvidia": "single stocks aren't in the model yet",
    "payroll": "payrolls aren't in the model (and revisions aren't handled)",
    "jobs": "payrolls aren't in the model (and revisions aren't handled)",
}


def expected_rejection(question: NumericQuestion) -> str | None:
    lowered = f"{question.question_text} {question.group_question_option or ''}".lower()
    for word, why in MUST_REJECT.items():
        if word in lowered:
            return why
    return None


async def closed_numeric_questions(
    client: MetaculusClient, tournament: str | int, limit: int
) -> list[NumericQuestion]:
    questions = await client.get_questions_matching_filter(
        ApiFilter(
            allowed_types=["numeric", "discrete"],
            allowed_statuses=["closed", "resolved"],
            allowed_tournaments=[tournament],
            group_question_mode="unpack_subquestions",
        ),
        num_questions=limit,
        error_if_question_target_missed=False,
    )
    # Belt and braces: never touch a question that is still open.
    now = datetime.now(timezone.utc)
    return [
        q for q in questions
        if isinstance(q, NumericQuestion) and q.close_time is not None and q.close_time <= now
    ]


def stand_in_ai_forecast(question: NumericQuestion) -> ReasonedPrediction[NumericDistribution]:
    """A plain AI-like forecast spread across the question's range."""
    span = question.upper_bound - question.lower_bound
    percentiles = [
        Percentile(percentile=p, value=question.lower_bound + span * (0.1 + 0.8 * p))
        for p in (0.1, 0.3, 0.5, 0.7, 0.9)
    ]
    return ReasonedPrediction(
        prediction_value=NumericDistribution.from_question(percentiles, question),
        reasoning="stand-in",
    )


def dry_run_blend(bot: FallTemplateBot2026, question: NumericQuestion, match: MatchResult) -> tuple[bool, str]:
    prediction = stand_in_ai_forecast(question)
    blended = bot._blend_with_numeric_model(question, prediction, match)
    if blended is prediction:
        return False, "blend FAILED: fell back to the AI forecast (see the warning above)"
    cdf = blended.prediction_value.get_cdf()
    heights = np.array([p.percentile for p in cdf])
    values = np.array([p.value for p in cdf])
    median = float(np.interp(0.5, heights, values))
    return True, f"blend OK at weight {os.environ['NUMERIC_MODEL_WEIGHT']}: blended median {median:,.4g}"


def resolution_value(resolution: str | None) -> float | None:
    try:
        return float(resolution) if resolution is not None else None
    except ValueError:
        return None  # e.g. "above_upper_bound" or "annulled"


def realized_outcome(result: MatchResult) -> float | None:
    """
    What actually happened, in the question's units, read from the series
    itself. None if the series hasn't published the end date yet.
    """
    assert result.proposal is not None
    proposal = result.proposal
    series = fetch_fred(proposal.series_id)  # cached: the full history
    if series.last_date < proposal.end_date:
        return None

    def value_on(day: date) -> float:
        index = int(np.searchsorted(series.dates, np.datetime64(day, "D"), side="right")) - 1
        return float(series.values[index])

    if proposal.quantity == "value":
        value = value_on(proposal.end_date)
    elif proposal.quantity in ("max", "min"):
        start = (
            np.datetime64(proposal.window_start, "D")
            if proposal.window_start and not proposal.all_time
            else series.dates[0]
        )
        window = series.values[(series.dates >= start) & (series.dates <= np.datetime64(proposal.end_date, "D"))]
        value = float(window.max() if proposal.quantity == "max" else window.min())
    else:
        assert proposal.reference_date is not None
        start_value, end_value = value_on(proposal.reference_date), value_on(proposal.end_date)
        if proposal.quantity == "pct_change":
            return (end_value / start_value - 1.0) * 100.0
        value = end_value - start_value
    return value * proposal.scale


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournaments", default="")
    parser.add_argument("--limit", type=int, default=300, help="most questions per tournament")
    args = parser.parse_args()

    # The blend dry run needs a weight above 0; this only affects this script.
    if not os.getenv("NUMERIC_MODEL_WEIGHT", "").strip():
        os.environ["NUMERIC_MODEL_WEIGHT"] = "0.5"

    client = MetaculusClient()
    tournaments: list[str | int] = [
        int(t) if t.strip().isdigit() else t.strip()
        for t in args.tournaments.split(",") if t.strip()
    ] or [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]

    questions: list[NumericQuestion] = []
    seen: set[int | None] = set()
    for tournament in tournaments:
        try:
            found = await closed_numeric_questions(client, tournament, args.limit)
        except Exception as error:
            print(f"Couldn't load tournament {tournament}: {error}")
            continue
        new = [q for q in found if q.id_of_question not in seen]
        seen.update(q.id_of_question for q in new)
        print(f"{tournament}: {len(new)} closed numeric questions")
        questions += new
    print(f"\nChecking {len(questions)} questions\n")

    llms = pick_models()
    helper = llms["parser"]
    bot = FallTemplateBot2026(publish_reports_to_metaculus=False, llms=llms)

    async def check(question: NumericQuestion) -> MatchResult:
        opened = question.open_time or question.published_time or datetime.now(timezone.utc)
        # Data only up to the day before the question opened: that day's
        # close may not have been known yet.
        as_of = opened.date() - timedelta(days=1)
        return await match_question(
            question_info(question), helper.invoke, as_of=as_of, require_criteria=False
        )

    results = await asyncio.gather(*(check(q) for q in questions), return_exceptions=True)

    accepted_lines: list[str] = []
    rejected_lines: list[str] = []
    reject_reasons: Counter[str] = Counter()
    wrong_accepts = blend_failures = errors = 0
    pits: list[float] = []
    pits_by_series: dict[str, list[float]] = defaultdict(list)
    by_series: Counter[str] = Counter()
    for question, result in zip(questions, results):
        title = question.question_text
        if question.group_question_option:
            title += f" [subquestion: {question.group_question_option}]"
        if isinstance(result, BaseException):
            errors += 1
            rejected_lines.append(f"ERROR  {title}: {result}")
            continue
        if not result.accepted:
            reason = "; ".join(result.reasons)
            reject_reasons[reason.split(":")[0][:60]] += 1
            rejected_lines.append(f"reject {title} ({question.page_url}): {reason}")
            continue

        assert result.proposal is not None and result.forecast is not None
        by_series[result.proposal.series_id] += 1
        lines = [f"ACCEPT {title}", f"       {question.page_url}"]
        why_reject = expected_rejection(question)
        if why_reject:
            wrong_accepts += 1
            lines.append(f"       <-- WRONG: should be rejected ({why_reject})")
        lines.append(f"       {'; '.join(result.reasons)}")
        criteria = " ".join((question.resolution_criteria or "").split())
        if criteria:
            lines.append(f"       resolves on: {criteria[:240]}{'...' if len(criteria) > 240 else ''}")
        else:
            lines.append("       resolves on: (not available: matched on the title alone)")
        lines.append(
            f"       unit {question.unit_of_measure!r}; range {question.lower_bound:,.4g} to "
            f"{question.upper_bound:,.4g}; opened {question.open_time:%Y-%m-%d}"
            if question.open_time else f"       unit {question.unit_of_measure!r}"
        )
        lines += ["       | " + line for line in anchor_text(result, question.unit_of_measure or "").splitlines()]
        value = resolution_value(question.resolution_string)
        source = "resolved"
        if value is None:
            value, source = realized_outcome(result), f"{result.proposal.series_id} was"
        if value is not None:
            pit = float(np.mean(result.scaled_samples <= value))
            pits.append(pit)
            pits_by_series[result.proposal.series_id].append(pit)
            lines.append(f"       {source} {value:,.4g}: the model's {pit:.0%} percentile")
        else:
            lines.append("       outcome not published yet")
        try:
            ok, text = dry_run_blend(bot, question, result)
        except Exception as error:
            ok, text = False, f"blend dry run crashed: {error}"
        if not ok:
            blend_failures += 1
        lines.append(f"       {text}")
        accepted_lines.append("\n".join(lines))

    groups = [q for q in questions if q.group_question_option]
    unlabelled = [
        q for q in groups
        if q.group_question_option.strip().lower() not in q.question_text.lower()
    ]
    summary = [
        f"{len(accepted_lines)} of {len(questions)} accepted ({dict(by_series)}); "
        f"{wrong_accepts} accepted that should have been rejected; "
        f"{blend_failures} blend failures; {errors} errors.",
    ]
    with_criteria = sum(1 for q in questions if (q.resolution_criteria or "").strip())
    summary.append(
        f"{with_criteria} of {len(questions)} questions came with resolution criteria "
        "(Metaculus withholds them for closed questions this account didn't forecast), "
        "so the rest were matched on titles alone."
    )
    if pits:
        def coverage(values: list[float]) -> str:
            array = np.array(values)
            inside80 = np.mean((array >= 0.1) & (array <= 0.9))
            inside90 = np.mean((array >= 0.05) & (array <= 0.95))
            return f"{inside80:.0%} inside the 80% range, {inside90:.0%} inside the 90% range"

        summary.append(
            f"Out-of-sample check on {len(pits)} matches with a known outcome "
            f"(data up to the day before each question opened): {coverage(pits)}."
        )
        for series_id, values in sorted(pits_by_series.items()):
            summary.append(f"  {series_id}: {len(values)} questions, {coverage(values)}")
    summary.append(
        f"Group subquestions: {len(groups)}, of which {len(unlabelled)} have a title "
        f"that doesn't contain their label"
        + (f" (e.g. {unlabelled[0].question_text!r} / {unlabelled[0].group_question_option!r})" if unlabelled else "")
        + "."
    )
    summary.append("Most common rejection reasons: " + "; ".join(
        f"{count}x {reason}" for reason, count in reject_reasons.most_common(6)
    ))

    report = "\n\n".join(accepted_lines) + "\n\n" + "\n".join(rejected_lines)
    print(report + "\n\n" + "\n".join(summary))
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as handle:
            handle.write("## Numeric model matching on closed questions\n\n")
            handle.write("\n\n".join(summary) + "\n\n```\n" + report + "\n```\n")
    return 1 if wrong_accepts or blend_failures else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(asyncio.run(main()))
