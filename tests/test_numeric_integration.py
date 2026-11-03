"""
Offline tests for connecting the numeric model to the bot: matching a
question to a series, the checks on that match, and blending distributions.
Uses synthetic data and stand-ins for the AI, so no network or API keys.

The last test class needs forecasting-tools and is skipped without it (it runs
in GitHub Actions, where the bot's dependencies are installed).

Run from the repository root with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import asyncio
import json
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numeric_model import Series  # noqa: E402
from numeric_model.blend import (  # noqa: E402
    MIN_CDF_STEP,
    blend_cdfs,
    check_cdf,
    max_pmf,
    model_cdf_on_grid,
    standardize_model_cdf,
)
from numeric_model.matching import (  # noqa: E402
    MatchResult,
    ProposedMatch,
    QuestionInfo,
    anchor_text,
    match_question,
    parse_matching_answer,
    units_compatible,
    verify_match,
)

TODAY = date(2026, 11, 3)


def business_days_until(end: date, count: int) -> np.ndarray:
    days = np.busday_offset(np.datetime64(end, "D"), -np.arange(count)[::-1], roll="backward")
    return days.astype("datetime64[D]")


def synthetic_series(series_id: str, level: float, vol: float, end: date = TODAY - timedelta(days=1), n: int = 3000, seed: int = 1) -> Series:
    rng = np.random.default_rng(seed)
    logs = np.log(level) + np.cumsum(rng.normal(0.0, vol, n))
    logs = logs - logs[-1] + np.log(level)  # end exactly at `level`
    return Series(series_id, business_days_until(end, n), np.exp(logs))


def weekly_series(series_id: str, level: float, end: date = TODAY - timedelta(days=1), n: int = 800) -> Series:
    rng = np.random.default_rng(2)
    days = np.array([np.datetime64(end - timedelta(weeks=n - 1 - i), "D") for i in range(n)])
    values = level + np.cumsum(rng.normal(0.0, 0.03, n))
    values = values - values[-1] + level
    return Series(series_id, days, values)


FAKE_DATA = {
    "SP500": synthetic_series("SP500", 6800.0, 0.01),
    "VIXCLS": synthetic_series("VIXCLS", 17.0, 0.05),
    "DGS10": synthetic_series("DGS10", 4.1, 0.01),
    "GASREGW": weekly_series("GASREGW", 3.10),
}


def fake_fetch(series_id: str) -> Series:
    if series_id not in FAKE_DATA:
        raise KeyError(series_id)
    return FAKE_DATA[series_id]


def sp500_question(**changes: object) -> QuestionInfo:
    fields = dict(
        title="What will the S&P 500 close at on December 31, 2026?",
        background="The S&P 500 is a stock market index.",
        resolution_criteria="Resolves to the S&P 500 closing value on December 31, 2026, as reported by FRED (SP500).",
        fine_print="",
        unit_of_measure="points",
        lower_bound=5500.0,
        upper_bound=8000.0,
        open_lower_bound=True,
        open_upper_bound=True,
        open_time=datetime(2026, 11, 1, tzinfo=timezone.utc),
        scheduled_resolution_time=datetime(2027, 1, 2, tzinfo=timezone.utc),
    )
    fields.update(changes)
    return QuestionInfo(**fields)  # type: ignore[arg-type]


def sp500_proposal(**changes: object) -> ProposedMatch:
    fields = dict(
        series_id="SP500",
        quantity="value",
        end_date=date(2026, 12, 31),
        window_start=None,
        reference_date=None,
        all_time=False,
        scale=1.0,
        source_match="exact",
        reason="The question resolves on the S&P 500 close.",
    )
    fields.update(changes)
    return ProposedMatch(**fields)  # type: ignore[arg-type]


def verify(question: QuestionInfo, proposal: ProposedMatch) -> MatchResult:
    return verify_match(question, proposal, TODAY, fetch=fake_fetch, n_paths=2000)


class ParseMatchingAnswerTest(unittest.TestCase):
    def test_reads_a_match_wrapped_in_text(self) -> None:
        answer = 'Sure: {"match": true, "series_id": "sp500", "quantity": "Value", "end_date": "2026-12-31", "window_start": null, "reference_date": null, "all_time": false, "scale": 1, "source_match": "exact", "reason": "x"} done'
        proposal = parse_matching_answer(answer)
        assert proposal is not None
        self.assertEqual(proposal.series_id, "SP500")
        self.assertEqual(proposal.quantity, "value")
        self.assertEqual(proposal.end_date, date(2026, 12, 31))
        self.assertIsNone(proposal.window_start)

    def test_no_match(self) -> None:
        self.assertIsNone(parse_matching_answer('{"match": false, "reason": "a poll"}'))

    def test_bad_answers_raise(self) -> None:
        for answer in ("no json here", '{"match": true}', '{"match": true, "end_date": "soon"}',
                       '{"match": true, "end_date": "2026-12-31", "scale": "100"}'):
            with self.assertRaises(ValueError, msg=answer):
                parse_matching_answer(answer)


class UnitsCompatibleTest(unittest.TestCase):
    def test_matching_units(self) -> None:
        self.assertTrue(units_compatible("index points", "value", 1.0, "points"))
        self.assertTrue(units_compatible("percent", "value", 1.0, "%"))
        self.assertTrue(units_compatible("US dollars per gallon", "value", 1.0, "$/gallon"))
        self.assertTrue(units_compatible("US dollars per gallon", "value", 100.0, "cents per gallon"))
        self.assertTrue(units_compatible("percent", "change", 100.0, "basis points"))
        self.assertTrue(units_compatible("index points", "pct_change", 1.0, "%"))
        self.assertTrue(units_compatible("index points", "value", 1.0, ""))

    def test_mismatched_units(self) -> None:
        # Dollars read as cents, or the other way round, is the classic unit error.
        self.assertFalse(units_compatible("US dollars per gallon", "value", 1.0, "cents per gallon"))
        self.assertFalse(units_compatible("US dollars per gallon", "value", 100.0, "$/gallon"))
        self.assertFalse(units_compatible("US dollars per barrel", "value", 1.0, "$ per gallon"))
        self.assertFalse(units_compatible("index points", "value", 1.0, "USD"))
        self.assertFalse(units_compatible("index points", "pct_change", 1.0, "points"))
        self.assertFalse(units_compatible("percent", "change", 1.0, "basis points"))
        self.assertFalse(units_compatible("index points", "value", 100.0, ""))


class VerifyMatchTest(unittest.TestCase):
    def test_accepts_a_good_match(self) -> None:
        result = verify(sp500_question(), sp500_proposal())
        self.assertTrue(result.accepted, result.reasons)
        assert result.forecast is not None
        self.assertGreater(result.forecast.steps, 30)
        self.assertLess(result.scaled_quantile(0.1), 6800)
        self.assertGreater(result.scaled_quantile(0.9), 6800)

    def assert_rejected(self, result: MatchResult, words: str) -> None:
        self.assertFalse(result.accepted)
        self.assertIn(words, " ".join(result.reasons))

    def test_rejects_unknown_series_and_proxies(self) -> None:
        self.assert_rejected(verify(sp500_question(), sp500_proposal(series_id="NVDA")), "not a known series")
        self.assert_rejected(verify(sp500_question(), sp500_proposal(source_match="proxy")), "proxy")

    def test_rejects_intraday_questions(self) -> None:
        question = sp500_question(title="What will the VIX's highest intraday value be between Oct 1 and Dec 24, 2026?")
        proposal = sp500_proposal(series_id="VIXCLS", quantity="max", window_start=date(2026, 10, 1))
        self.assert_rejected(verify(question, proposal), "intraday")

    def test_rejects_bad_quantities_and_scales(self) -> None:
        self.assert_rejected(verify(sp500_question(), sp500_proposal(quantity="average")), "unknown quantity")
        self.assert_rejected(verify(sp500_question(), sp500_proposal(scale=2.0)), "power of ten")

    def test_rejects_bad_dates(self) -> None:
        self.assert_rejected(verify(sp500_question(), sp500_proposal(end_date=date(2026, 10, 1))), "before the forecast date")
        self.assert_rejected(verify(sp500_question(), sp500_proposal(end_date=date(2027, 3, 1))), "after the resolution date")
        self.assert_rejected(verify(sp500_question(), sp500_proposal(quantity="max")), "window start")
        self.assert_rejected(verify(sp500_question(), sp500_proposal(quantity="max", window_start=date(2027, 1, 5))), "starts after")
        self.assert_rejected(verify(sp500_question(unit_of_measure="%"), sp500_proposal(quantity="pct_change")), "reference date")

    def test_rejects_wrong_units(self) -> None:
        self.assert_rejected(verify(sp500_question(unit_of_measure="USD"), sp500_proposal()), "doesn't fit")

    def test_rejects_implausible_values(self) -> None:
        # Gas at $3.10 against a question in cents (range 250-400) with scale 1
        # would be caught by the unit check; with an empty unit it is caught here.
        question = sp500_question(
            title="What will the US regular gasoline price be on December 28, 2026?",
            resolution_criteria="EIA weekly US regular gasoline retail price.",
            unit_of_measure="",
            lower_bound=250.0,
            upper_bound=400.0,
        )
        proposal = sp500_proposal(series_id="GASREGW", end_date=date(2026, 12, 28))
        self.assert_rejected(verify(question, proposal), "far outside")

    def test_rejects_stale_or_missing_data(self) -> None:
        old = synthetic_series("SP500", 6800.0, 0.01, end=TODAY - timedelta(days=40))
        result = verify_match(sp500_question(), sp500_proposal(), TODAY, fetch=lambda _: old, n_paths=500)
        self.assert_rejected(result, "days old")

        def broken(_: str) -> Series:
            raise ConnectionError("FRED is down")

        result = verify_match(sp500_question(), sp500_proposal(), TODAY, fetch=broken)
        self.assert_rejected(result, "couldn't download")

    def test_cuts_data_at_the_forecast_date(self) -> None:
        # Testing on a closed question: data after the open date must not be used.
        as_of = TODAY - timedelta(days=60)
        question = sp500_question(open_time=datetime.combine(as_of, datetime.min.time(), tzinfo=timezone.utc))
        result = verify_match(question, sp500_proposal(), as_of, fetch=fake_fetch, n_paths=500)
        self.assertTrue(result.accepted, result.reasons)
        assert result.forecast is not None
        self.assertLessEqual(result.forecast.last_date, as_of)

    def test_cents_question_with_scale_100(self) -> None:
        question = sp500_question(
            title="What will the US regular gasoline price be on December 28, 2026?",
            resolution_criteria="EIA weekly US regular gasoline retail price.",
            unit_of_measure="cents per gallon",
            lower_bound=250.0,
            upper_bound=400.0,
        )
        proposal = sp500_proposal(series_id="GASREGW", end_date=date(2026, 12, 28), scale=100.0)
        result = verify(question, proposal)
        self.assertTrue(result.accepted, result.reasons)
        self.assertAlmostEqual(result.scaled_quantile(0.5), 310, delta=30)
        text = anchor_text(result, "cents per gallon")
        self.assertIn("cents per gallon", text)
        self.assertIn("310", text)  # latest value in cents


class MatchQuestionTest(unittest.TestCase):
    def run_match(self, answer: str) -> MatchResult:
        prompts: list[str] = []

        async def ask(prompt: str) -> str:
            prompts.append(prompt)
            return answer

        result = asyncio.run(match_question(sp500_question(), ask, TODAY, fetch=fake_fetch))
        self.assertEqual(len(prompts), 1)
        self.assertIn("SP500", prompts[0])
        self.assertIn("S&P 500 close at on December 31", prompts[0])
        return result

    def test_end_to_end(self) -> None:
        answer = json.dumps(dict(match=True, series_id="SP500", quantity="value", end_date="2026-12-31",
                                 window_start=None, reference_date=None, all_time=False, scale=1,
                                 source_match="exact", reason="S&P close"))
        self.assertTrue(self.run_match(answer).accepted)

    def test_no_match_and_garbage(self) -> None:
        self.assertFalse(self.run_match('{"match": false}').accepted)
        self.assertFalse(self.run_match("I don't know").accepted)


def ai_like_cdf(grid: np.ndarray, center: float, width: float, open_lower: bool, open_upper: bool) -> np.ndarray:
    """A stand-in for forecasting-tools' get_cdf(): a logistic CDF, standardized
    with exactly the library's formula (1% spread, so flat parts rise by the
    bare minimum step)."""
    raw = 1.0 / (1.0 + np.exp(-(grid - center) / width))
    if not open_lower:
        raw = (raw - raw[0]) / (raw[-1] - raw[0])
    if not open_upper:
        raw = raw / raw[-1]
    location = np.linspace(0, 1, len(grid))
    low_to = 0.0 if open_lower else raw[0]
    high_to = 1.0 if open_upper else raw[-1]
    r = (raw - low_to) / (high_to - low_to)
    if open_lower and open_upper:
        return 0.988 * r + 0.01 * location + 0.001
    if open_lower:
        return 0.989 * r + 0.01 * location + 0.001
    if open_upper:
        return 0.989 * r + 0.01 * location
    return 0.99 * r + 0.01 * location


class BlendTest(unittest.TestCase):
    grid = np.linspace(0.0, 100.0, 201)

    def test_model_cdf_on_grid(self) -> None:
        samples = np.array([-5.0, 10.0, 10.0, 50.0, 120.0])
        open_cdf = model_cdf_on_grid(samples, self.grid, True, True)
        self.assertAlmostEqual(open_cdf[0], 0.2)  # P(< lower bound)
        self.assertAlmostEqual(open_cdf[20], 0.6)  # P(<= 10)
        self.assertAlmostEqual(open_cdf[-1], 0.8)  # P(<= upper bound)
        closed = model_cdf_on_grid(samples, self.grid, False, False)
        self.assertEqual(closed[0], 0.0)  # nothing below a closed lower bound
        self.assertAlmostEqual(closed[1], 0.2)  # the -5 sample lands on the bound
        self.assertEqual(closed[-1], 1.0)

    def test_blends_meet_metaculus_rules_for_every_bound_type(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.normal(70.0, 8.0, 5000)
        for open_lower in (True, False):
            for open_upper in (True, False):
                ai = ai_like_cdf(self.grid, 40.0, 10.0, open_lower, open_upper)
                check_cdf(ai, open_lower, open_upper)
                for weight in (0.01, 0.3, 0.5, 1.0):
                    blended = blend_cdfs(ai, self.grid, samples, weight, open_lower, open_upper)
                    check_cdf(blended, open_lower, open_upper)
                    self.assertGreater(np.diff(blended).min(), MIN_CDF_STEP)

    def test_weights(self) -> None:
        samples = np.random.default_rng(1).normal(70.0, 8.0, 5000)
        ai = ai_like_cdf(self.grid, 40.0, 10.0, True, True)
        np.testing.assert_allclose(blend_cdfs(ai, self.grid, samples, 0.0, True, True), ai)
        half = blend_cdfs(ai, self.grid, samples, 0.5, True, True)
        median_index = int(np.searchsorted(half, 0.5))
        # Halfway between the AI's median (40) and the model's (70).
        self.assertTrue(40 < self.grid[median_index] < 70)
        with self.assertRaises(ValueError):
            blend_cdfs(ai, self.grid, samples, 1.5, True, True)

    def test_near_certain_model_is_capped(self) -> None:
        # E.g. the end date has passed, so every sample is the same value.
        samples = np.full(5000, 55.0)
        model = standardize_model_cdf(model_cdf_on_grid(samples, self.grid, False, False), False, False)
        self.assertLessEqual(np.diff(model).max(), 0.9 * max_pmf(201) + 1e-12)
        ai = ai_like_cdf(self.grid, 40.0, 10.0, False, False)
        check_cdf(blend_cdfs(ai, self.grid, samples, 1.0, False, False), False, False)

    def test_discrete_grid(self) -> None:
        # A discrete question with outcomes 0..10: bucket edges at -0.5, 0.5, ... 10.5.
        grid = np.arange(-0.5, 11.0, 1.0)
        samples = np.random.default_rng(2).normal(6.2, 1.0, 5000)
        ai = ai_like_cdf(grid, 4.0, 1.5, False, False)
        blended = blend_cdfs(ai, grid, samples, 0.5, False, False)
        check_cdf(blended, False, False)
        model_only = model_cdf_on_grid(samples, grid, False, False)
        # Samples between 5.5 and 6.5 count as the outcome 6.
        expected = np.mean((samples > 5.5) & (samples <= 6.5))
        self.assertAlmostEqual(model_only[7] - model_only[6], expected)


try:
    from forecasting_tools import NumericDistribution, NumericQuestion, Percentile

    HAVE_FORECASTING_TOOLS = True
except ImportError:
    HAVE_FORECASTING_TOOLS = False


@unittest.skipUnless(HAVE_FORECASTING_TOOLS, "forecasting-tools is not installed")
class ForecastingToolsRoundTripTest(unittest.TestCase):
    """The blended CDF must survive forecasting-tools' own validation, which
    is what Metaculus will see."""

    def make_question(self, open_lower: bool, open_upper: bool, zero_point: float | None = None) -> "NumericQuestion":
        return NumericQuestion(
            question_text="What will the S&P 500 close at?",
            lower_bound=5500.0,
            upper_bound=8000.0,
            open_lower_bound=open_lower,
            open_upper_bound=open_upper,
            zero_point=zero_point,
            unit_of_measure="points",
        )

    def test_round_trip(self) -> None:
        samples = np.random.default_rng(3).normal(7200.0, 300.0, 5000)
        for open_lower in (True, False):
            for open_upper in (True, False):
                for zero_point in (None, 0.0):
                    question = self.make_question(open_lower, open_upper, zero_point)
                    ai = NumericDistribution.from_question(
                        [Percentile(percentile=p, value=v) for p, v in
                         ((0.1, 6200), (0.3, 6600), (0.5, 6800), (0.7, 7000), (0.9, 7400))],
                        question,
                    )
                    ai_cdf = ai.get_cdf()
                    grid = np.array([p.value for p in ai_cdf])
                    heights = np.array([p.percentile for p in ai_cdf])
                    for weight in (0.2, 0.5, 1.0):
                        blended = blend_cdfs(heights, grid, samples, weight, open_lower, open_upper)
                        distribution = NumericDistribution.from_question(
                            [Percentile(value=float(v), percentile=float(h)) for v, h in zip(grid, blended)],
                            question,
                        )
                        final = distribution.get_cdf()
                        final_heights = np.array([p.percentile for p in final])
                        median = grid[int(np.searchsorted(final_heights, 0.5))]
                        self.assertTrue(6700 <= median <= 7300, (open_lower, open_upper, zero_point, weight, median))


if __name__ == "__main__":
    unittest.main()
