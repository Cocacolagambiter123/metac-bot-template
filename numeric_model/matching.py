"""
Matches a Metaculus numeric question to one of the series in KNOWN_SERIES,
then checks the match in code before the bot is allowed to use it.

A cheap AI call reads the question and proposes a match (series, quantity,
dates, unit conversion). Nothing it says is trusted on its own: every field is
checked here, the series is downloaded, and the model's forecast has to be
plausible against the question's bounds. Any doubt rejects the match, because
a wrong match (e.g. the wrong series, or dollars read as cents) is worse than
no match at all.

This module has no dependency on forecasting-tools, so it can be tested
offline. main.py turns a forecasting-tools question into a QuestionInfo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np

from numeric_model.data_sources import Series, fetch_fred
from numeric_model.registry import KNOWN_SERIES
from numeric_model.simulate import ForecastResult, QuestionSpec, forecast

logger = logging.getLogger(__name__)

QUANTITIES = ("value", "max", "min", "change", "pct_change")
# Unit conversions allowed between the series and the question, e.g. 100 for
# a series in dollars and a question in cents. Anything else is rejected.
ALLOWED_SCALES = (1e-6, 1e-3, 1e-2, 1.0, 1e2, 1e3, 1e6)


@dataclass(frozen=True)
class QuestionInfo:
    """The parts of a Metaculus numeric question that matching needs."""

    title: str  # the question itself, e.g. "What will the VIX close at on ...?"
    background: str
    resolution_criteria: str
    fine_print: str
    unit_of_measure: str
    lower_bound: float
    upper_bound: float
    open_lower_bound: bool
    open_upper_bound: bool
    is_discrete: bool = False
    open_time: datetime | None = None
    scheduled_resolution_time: datetime | None = None
    page_url: str = ""
    # For one subquestion of a group question: its label, e.g. "Nasdaq-100"
    # in a group about several indices. The title alone may not say which.
    group_option: str = ""


@dataclass(frozen=True)
class ProposedMatch:
    """What the AI proposed, parsed from its JSON answer."""

    series_id: str
    quantity: str
    end_date: date
    window_start: date | None
    reference_date: date | None
    all_time: bool
    scale: float
    source_match: str
    reason: str


@dataclass
class MatchResult:
    """The outcome of matching. `accepted` is True only if every check passed."""

    accepted: bool
    reasons: list[str] = field(default_factory=list)
    proposal: ProposedMatch | None = None
    forecast: ForecastResult | None = None

    @property
    def scaled_samples(self) -> np.ndarray:
        """The model's samples in the question's units."""
        assert self.forecast is not None and self.proposal is not None
        return self.forecast.samples * self.proposal.scale

    def scaled_quantile(self, q: float) -> float:
        return float(np.quantile(self.scaled_samples, q))

    def describe(self) -> str:
        status = "accepted" if self.accepted else "rejected"
        return f"Numeric model match {status}: " + "; ".join(self.reasons)


# --------------------------------------------------------------------------
# The AI's part: a prompt and a parser for its answer.
# --------------------------------------------------------------------------


def build_matching_prompt(question: QuestionInfo, today: date) -> str:
    menu = "\n".join(
        f"- {s.series_id}: {s.description} (units: {s.units})"
        for s in KNOWN_SERIES.values()
    )
    resolution = (
        question.scheduled_resolution_time.date().isoformat()
        if question.scheduled_resolution_time
        else "unknown"
    )
    subquestion = (
        f'\nThis is one subquestion of a group of related questions. It is only about: "{question.group_option}". Match only that subquestion; the title and background may describe the whole group.'
        if question.group_option
        else ""
    )
    return f"""
You decide whether a forecasting question is about one of a fixed list of data series from FRED (the St. Louis Fed's data site). Your answer is checked by a program, so follow the format exactly.

Available series:
{menu}

Question: {question.title}{subquestion}
Background: {question.background}
Resolution criteria: {question.resolution_criteria}
Fine print: {question.fine_print}
Unit of the answer: {question.unit_of_measure or "not stated"}
Answer range set by the question: {question.lower_bound} to {question.upper_bound}
Scheduled resolution date: {resolution}
Today: {today.isoformat()}

Rules:
- Only match if the question resolves on exactly the quantity a listed series measures. If it resolves on a different source that only approximates the series (e.g. AAA's gas price instead of the EIA figures, a regional price instead of the national one, an intraday high instead of the daily close, a single stock instead of an index, or a monthly average instead of a daily value), set "source_match" to "proxy".
- "source_match" is "exact" only if the question's resolution source publishes the same numbers as the series (e.g. the S&P 500 close, the CBOE VIX close, EIA's weekly US regular gasoline price, Treasury's 10-year yield).
- "quantity" is one of: "value" (the value on the end date), "max" or "min" (the highest or lowest value from window_start to end_date), "change" (end value minus the value on reference_date, in the series' units), "pct_change" (percentage change from reference_date to end_date).
- "end_date" is the last date whose data counts, which can be earlier than the resolution date.
- For "max"/"min", give "window_start", or set "all_time" to true if the question asks about an all-time high or low.
- For "change"/"pct_change", give "reference_date".
- "scale" converts the series' units to the question's units: 1 if they are the same, 100 if the series is in dollars and the question in cents, 0.001 if the question is in thousands of the series' unit, and so on. Use only powers of ten.
- If you are not sure, answer with "match": false. A wrong match is worse than no match.

Answer with only one JSON object, no other text:
{{"match": true or false, "series_id": "...", "quantity": "...", "end_date": "YYYY-MM-DD", "window_start": "YYYY-MM-DD" or null, "reference_date": "YYYY-MM-DD" or null, "all_time": true or false, "scale": 1, "source_match": "exact" or "proxy", "reason": "one sentence"}}
""".strip()


def _parse_date(raw: object) -> date | None:
    if raw is None or raw == "":
        return None
    if not isinstance(raw, str):
        raise ValueError(f"Not a date: {raw!r}")
    return date.fromisoformat(raw.strip()[:10])


def parse_matching_answer(text: str) -> ProposedMatch | None:
    """
    Parses the AI's JSON answer. Returns None if it said there is no match.
    Raises ValueError if the answer can't be read.
    """
    found = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not found:
        raise ValueError("No JSON object in the answer")
    data = json.loads(found.group(0))
    if not isinstance(data, dict):
        raise ValueError("The answer is not a JSON object")
    if data.get("match") is not True:
        return None
    end_date = _parse_date(data.get("end_date"))
    if end_date is None:
        raise ValueError("No end_date")
    scale = data.get("scale", 1)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"scale is not a number: {scale!r}")
    return ProposedMatch(
        series_id=str(data.get("series_id", "")).strip().upper(),
        quantity=str(data.get("quantity", "")).strip().lower(),
        end_date=end_date,
        window_start=_parse_date(data.get("window_start")),
        reference_date=_parse_date(data.get("reference_date")),
        all_time=data.get("all_time") is True,
        scale=float(scale),
        source_match=str(data.get("source_match", "")).strip().lower(),
        reason=str(data.get("reason", "")).strip(),
    )


# --------------------------------------------------------------------------
# The program's part: checks that don't trust the AI.
# --------------------------------------------------------------------------

# Words that mean the question is about something the model can't see in
# daily closing data.
_UNSUPPORTED_WORDS = ("intraday", "intra-day", "during the trading day", "tick")

# For each series unit: words that may appear in the question's unit, by the
# scale between them. An empty question unit is allowed only at scale 1.
_PERCENT = ("%", "percent", "percentage", "percentage point", "pp")
_BASIS_POINTS = ("basis point", "bp")
_UNIT_WORDS: dict[str, dict[float, tuple[str, ...]]] = {
    "index points": {1.0: ("point", "index", "level", "value", "pt")},
    # A yield or spread of 4.2% is also written "4.2 pp", or "420 bp".
    "percent": {1.0: _PERCENT, 100.0: _BASIS_POINTS},
    "US dollars per barrel": {1.0: ("$", "usd", "dollar")},
    "US dollars per gallon": {1.0: ("$", "usd", "dollar"), 100.0: ("cent", "¢")},
    "US dollars": {1.0: ("$", "usd", "dollar"), 100.0: ("cent", "¢")},
    "yen": {1.0: ("yen", "¥", "jpy")},
}
# Changes in a percent series are in percentage points, or basis points at 100.
_PERCENT_CHANGE_WORDS: dict[float, tuple[str, ...]] = {1.0: _PERCENT, 100.0: _BASIS_POINTS}


def _unit_has(unit: str, word: str) -> bool:
    """
    Whether the question's unit contains `word`. Symbols such as "$" and "%"
    can appear anywhere; words must be whole words, optionally plural, so
    "pp" matches "pp" but not "ppm", and "cent" matches "cents" but not
    "century".
    """
    if not word[0].isalpha():
        return word in unit
    return re.search(rf"(?<![a-z]){re.escape(word)}s?(?![a-z])", unit) is not None


def units_compatible(series_units: str, quantity: str, scale: float, question_unit: str) -> bool:
    unit = question_unit.strip().lower()
    if quantity == "pct_change":
        return scale == 1.0 and any(_unit_has(unit, w) for w in ("%", "percent", "percentage"))
    if not unit:
        return scale == 1.0
    if quantity == "change" and series_units == "percent":
        table = _PERCENT_CHANGE_WORDS
    else:
        table = _UNIT_WORDS.get(series_units, {})
    words = table.get(scale)
    if not words:
        return False
    if any(_unit_has(unit, w) for w in words):
        # "%" alone must not pass for a dollar series, and a "per barrel"
        # question must not match a "per gallon" series.
        if "barrel" in series_units and "gallon" in unit:
            return False
        if "gallon" in series_units and "barrel" in unit:
            return False
        return True
    return False


def _max_staleness_days(series: Series) -> float:
    return max(10.0, 3 * series.median_step_days())


def verify_match(
    question: QuestionInfo,
    proposal: ProposedMatch,
    as_of: date,
    fetch: Callable[[str], Series] = fetch_fred,
    n_paths: int = 5000,
) -> MatchResult:
    """
    Checks a proposed match. `as_of` is the date the forecast is made: today
    in live use, or the question's open date when testing on closed questions
    (the series is then cut off at that date, so nothing later leaks in).
    """
    result = MatchResult(accepted=False, proposal=proposal)
    reject = result.reasons.append

    known = KNOWN_SERIES.get(proposal.series_id)
    if known is None:
        reject(f"{proposal.series_id!r} is not a known series")
        return result
    if proposal.source_match != "exact":
        reject(f"the question resolves on a proxy for {proposal.series_id}, not the series itself")
        return result
    text = " ".join(
        (question.title, question.group_option, question.resolution_criteria, question.fine_print)
    ).lower()
    for word in _UNSUPPORTED_WORDS:
        if word in text:
            reject(f"the question mentions {word!r}, but the model only has daily closes")
            return result
    if proposal.quantity not in QUANTITIES:
        reject(f"unknown quantity {proposal.quantity!r}")
        return result
    if proposal.scale not in ALLOWED_SCALES:
        reject(f"unit conversion {proposal.scale} is not an allowed power of ten")
        return result
    if proposal.quantity == "pct_change" and proposal.scale != 1.0:
        reject("a percentage change can't have a unit conversion")
        return result

    # Dates.
    if proposal.end_date < as_of:
        reject(f"end date {proposal.end_date} is before the forecast date {as_of}")
        return result
    if proposal.end_date > as_of + timedelta(days=3 * 366):
        reject(f"end date {proposal.end_date} is more than 3 years away")
        return result
    if question.scheduled_resolution_time is not None:
        resolves = question.scheduled_resolution_time.date()
        if proposal.end_date > resolves + timedelta(days=7):
            reject(f"end date {proposal.end_date} is after the resolution date {resolves}")
            return result
    if question.open_time is not None and proposal.end_date < question.open_time.date():
        reject(f"end date {proposal.end_date} is before the question opened")
        return result
    if proposal.quantity in ("max", "min"):
        if proposal.window_start is None and not proposal.all_time:
            reject("a max/min question needs a window start or an all-time flag")
            return result
        if proposal.window_start is not None and proposal.window_start > proposal.end_date:
            reject("the window starts after it ends")
            return result
    if proposal.quantity in ("change", "pct_change"):
        if proposal.reference_date is None:
            reject("a change question needs a reference date")
            return result
        if proposal.reference_date >= proposal.end_date:
            reject("the reference date is not before the end date")
            return result

    # Units.
    if not units_compatible(known.units, proposal.quantity, proposal.scale, question.unit_of_measure):
        reject(
            f"the question's unit {question.unit_of_measure!r} doesn't fit {known.units!r} "
            f"at scale {proposal.scale:g} for {proposal.quantity!r}"
        )
        return result

    # Data.
    try:
        series = fetch(proposal.series_id)
    except Exception as error:  # network or parse failure
        reject(f"couldn't download {proposal.series_id}: {error}")
        return result
    try:
        series = series.up_to(as_of)
    except ValueError:
        reject(f"{proposal.series_id} has no data before {as_of}")
        return result
    stale_days = (as_of - series.last_date).days
    if stale_days > _max_staleness_days(series):
        reject(f"{proposal.series_id}'s latest data ({series.last_date}) is {stale_days} days old")
        return result

    spec = QuestionSpec(
        end_date=proposal.end_date,
        quantity=proposal.quantity,  # type: ignore[arg-type]
        window_start=None if proposal.all_time else proposal.window_start,
        reference_date=proposal.reference_date,
    )
    try:
        model_forecast = forecast(series, spec, known.config, n_paths=n_paths)
    except Exception as error:
        reject(f"the model failed: {error}")
        return result
    result.forecast = model_forecast

    # Plausibility against the question's bounds. Question writers set the
    # range around plausible outcomes, so a model median far outside it means
    # the match (or its units) is probably wrong.
    span = question.upper_bound - question.lower_bound
    if span <= 0:
        reject("the question's range is empty")
        return result
    median = result.scaled_quantile(0.5)
    low, high = question.lower_bound - 0.25 * span, question.upper_bound + 0.25 * span
    if not low <= median <= high:
        reject(
            f"the model's median {median:,.4g} is far outside the question's range "
            f"{question.lower_bound:,.4g} to {question.upper_bound:,.4g}"
        )
        return result
    if proposal.quantity in ("value", "max", "min"):
        current = series.last_value * proposal.scale
        if not question.lower_bound - span <= current <= question.upper_bound + span:
            reject(
                f"the latest value {current:,.4g} is far outside the question's range "
                f"{question.lower_bound:,.4g} to {question.upper_bound:,.4g}"
            )
            return result

    result.accepted = True
    result.reasons.append(
        f"{proposal.series_id} {proposal.quantity} by {proposal.end_date} "
        f"(scale {proposal.scale:g}); {proposal.reason}"
    )
    return result


def anchor_text(result: MatchResult, unit: str) -> str:
    """The model's forecast in the question's units, for the forecasting prompt."""
    assert result.accepted and result.forecast is not None and result.proposal is not None
    fc, prop = result.forecast, result.proposal
    known = KNOWN_SERIES[prop.series_id]
    unit = unit or known.units

    def q(level: float) -> str:
        return _fmt(result.scaled_quantile(level))

    vol_ratio = fc.current_vol / fc.long_run_vol if fc.long_run_vol else 1.0
    what = {
        "value": f"its value on {prop.end_date}",
        "max": f"its highest daily close from {prop.window_start or 'the start of the data'} to {prop.end_date}",
        "min": f"its lowest daily close from {prop.window_start or 'the start of the data'} to {prop.end_date}",
        "change": f"its change from {prop.reference_date} to {prop.end_date}",
        "pct_change": f"its percentage change from {prop.reference_date} to {prop.end_date}",
    }[prop.quantity]
    lines = [
        f"A statistical model fitted to the history of {known.description} (FRED series {prop.series_id}) "
        f"simulated {what}. Results in {unit}:",
        f"- Median {q(0.5)}; 80% range {q(0.1)} to {q(0.9)}; 90% range {q(0.05)} to {q(0.95)}.",
        f"- Latest observation: {_fmt(fc.last_value * prop.scale)} on {fc.last_date} "
        f"({fc.steps} more observations before the end date). Current volatility is "
        f"{vol_ratio:.2f}x its 10-year average.",
        "- In historical tests its 80% ranges held the outcome about 80% of the time. "
        "It only knows past movements of the series, not news, scheduled events or "
        "expert views, so use it as a base rate and adjust for what it can't see.",
    ]
    lines += [f"- {note}" for note in fc.notes]
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.0f}"


async def match_question(
    question: QuestionInfo,
    ask: Callable[[str], "object"],
    as_of: date,
    fetch: Callable[[str], Series] = fetch_fred,
) -> MatchResult:
    """
    Runs the whole match: asks the AI (`ask` is an async function taking a
    prompt and returning text), parses its answer and verifies it.
    """
    prompt = build_matching_prompt(question, as_of)
    try:
        answer = await ask(prompt)  # type: ignore[misc]
        proposal = parse_matching_answer(str(answer))
    except Exception as error:
        return MatchResult(accepted=False, reasons=[f"couldn't read the matching answer: {error}"])
    if proposal is None:
        return MatchResult(accepted=False, reasons=["the question isn't about a known series"])
    # Downloading and simulating take a few seconds, so keep them off the event loop.
    return await asyncio.to_thread(verify_match, question, proposal, as_of, fetch)
