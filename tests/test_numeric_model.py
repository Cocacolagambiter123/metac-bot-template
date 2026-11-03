"""
Offline tests for numeric_model (synthetic data, no network).

Run from the repository root with:  python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numeric_model import (  # noqa: E402
    ModelConfig,
    QuestionSpec,
    Series,
    forecast,
    parse_fred_csv,
    steps_between,
)


def business_days(start: date, count: int) -> np.ndarray:
    days = np.busday_offset(np.datetime64(start, "D"), np.arange(count), roll="forward")
    return days.astype("datetime64[D]")


def synthetic_random_walk(n: int, daily_vol: float, seed: int) -> Series:
    rng = np.random.default_rng(seed)
    log_prices = np.log(100.0) + np.cumsum(rng.normal(0.0, daily_vol, n))
    return Series("TEST", business_days(date(2000, 1, 3), n), np.exp(log_prices))


class ParseFredCsvTest(unittest.TestCase):
    def test_skips_missing_values_and_sorts(self) -> None:
        text = "observation_date,VIXCLS\n2024-01-03,13.5\n2024-01-01,.\n2024-01-02,12.0\n"
        series = parse_fred_csv("VIXCLS", text)
        self.assertEqual(list(series.values), [12.0, 13.5])
        self.assertEqual(series.last_date, date(2024, 1, 3))

    def test_accepts_old_header_style(self) -> None:
        series = parse_fred_csv("DGS10", "DATE,DGS10\n2024-01-02,3.95\n")
        self.assertEqual(series.last_value, 3.95)


class StepsBetweenTest(unittest.TestCase):
    def test_daily_series_counts_business_days(self) -> None:
        series = synthetic_random_walk(100, 0.01, seed=1)
        friday = date(2024, 1, 5)
        self.assertEqual(steps_between(series, friday, date(2024, 1, 8)), 1)  # Monday
        self.assertEqual(steps_between(series, friday, date(2024, 1, 7)), 0)  # Sunday
        self.assertEqual(steps_between(series, friday, date(2024, 1, 12)), 5)

    def test_weekly_series_counts_weeks(self) -> None:
        dates = np.arange(np.datetime64("2020-01-06"), np.datetime64("2024-01-01"), np.timedelta64(7, "D"))
        series = Series("WEEKLY", dates.astype("datetime64[D]"), np.ones(len(dates)))
        self.assertEqual(steps_between(series, date(2024, 1, 1), date(2024, 1, 29)), 4)


class ForecastTest(unittest.TestCase):
    def test_percentiles_are_ordered(self) -> None:
        series = synthetic_random_walk(1500, 0.01, seed=2)
        result = forecast(series, QuestionSpec(end_date=series.last_date + timedelta(days=30)))
        values = list(result.percentiles.values())
        self.assertEqual(values, sorted(values))

    def test_window_max_is_at_least_end_value_and_observed_max(self) -> None:
        series = synthetic_random_walk(1500, 0.01, seed=3)
        window_start = series.last_date - timedelta(days=60)
        spec = QuestionSpec(
            end_date=series.last_date + timedelta(days=30),
            quantity="max",
            window_start=window_start,
        )
        result = forecast(series, spec)
        observed_max = series.since(window_start).values.max()
        self.assertTrue(np.all(result.samples >= observed_max - 1e-9))
        end_only = forecast(series, QuestionSpec(end_date=spec.end_date))
        self.assertGreaterEqual(result.quantile(0.5), end_only.quantile(0.5))

    def test_mean_reversion_pulls_towards_long_run_level(self) -> None:
        rng = np.random.default_rng(4)
        n, phi, mean = 3000, 0.97, np.log(20.0)
        x = np.empty(n)
        x[0] = mean
        for t in range(1, n):
            x[t] = mean + phi * (x[t - 1] - mean) + rng.normal(0, 0.05)
        x[-1] = np.log(40.0)  # end far above the long-run level
        series = Series("MR", business_days(date(2000, 1, 3), n), np.exp(x))
        spec = QuestionSpec(end_date=series.last_date + timedelta(days=60))
        result = forecast(series, spec, ModelConfig(mean_reverting=True, history_years=None))
        self.assertLess(result.quantile(0.5), 35.0)

    def test_momentum_model_widens_ranges_for_persistent_moves(self) -> None:
        # Weekly changes with strong momentum (AR(1) on changes, rho = 0.6):
        # an iid random walk is too narrow over several weeks; momentum isn't.
        rng = np.random.default_rng(7)
        n, rho = 1500, 0.6
        changes = np.zeros(n)
        for t in range(1, n):
            changes[t] = rho * changes[t - 1] + rng.normal(0, 0.01)
        dates = np.arange(np.datetime64("1995-01-02"), np.datetime64("1995-01-02") + np.timedelta64(7 * n, "D"), np.timedelta64(7, "D"))
        series = Series("WEEKLY", dates.astype("datetime64[D]"), np.exp(np.log(3.0) + np.cumsum(changes)))
        spec = QuestionSpec(end_date=series.last_date + timedelta(weeks=8))
        plain = forecast(series, spec, ModelConfig(history_years=None))
        momentum = forecast(series, spec, ModelConfig(momentum=True, history_years=None))
        width = lambda r: r.percentiles[0.9] - r.percentiles[0.1]  # noqa: E731
        self.assertGreater(width(momentum), 1.5 * width(plain))

    def test_change_against_an_observed_reference(self) -> None:
        series = synthetic_random_walk(1500, 0.01, seed=5)
        spec = QuestionSpec(end_date=series.last_date + timedelta(days=14), quantity="change")
        result = forecast(series, spec)
        values = forecast(series, QuestionSpec(end_date=spec.end_date))
        np.testing.assert_allclose(result.samples, values.samples - series.last_value)

    def test_intervals_are_calibrated_on_a_known_random_walk(self) -> None:
        # Forecast 20 business days ahead from many origins; the true outcome
        # should land inside the 10th-90th percentile range about 80% of the time.
        series = synthetic_random_walk(4000, 0.01, seed=6)
        hits, total = 0, 0
        for origin in range(1000, 3900, 29):
            past = Series("TEST", series.dates[: origin + 1], series.values[: origin + 1])
            target_index = origin + 20
            spec = QuestionSpec(end_date=series.dates[target_index].astype(object))
            result = forecast(past, spec, n_paths=2000, seed=origin)
            outcome = series.values[target_index]
            hits += result.percentiles[0.1] <= outcome <= result.percentiles[0.9]
            total += 1
        coverage = hits / total
        self.assertGreater(coverage, 0.70, f"80% range covered only {coverage:.0%}")
        self.assertLess(coverage, 0.90, f"80% range covered {coverage:.0%}")


if __name__ == "__main__":
    unittest.main()
