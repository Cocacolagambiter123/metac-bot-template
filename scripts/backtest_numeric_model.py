"""
Historical calibration test for numeric_model.

From many past dates ("origins"), forecast each series several steps ahead
using only the data available on that date, then compare with what actually
happened. A well-calibrated model's 80% range (10th-90th percentile) should
contain the outcome about 80% of the time.

Each model is compared with a naive baseline: a normal random walk with a
constant volatility taken from the same history. The comparison uses CRPS (a
proper score for whole distributions; lower is better). "Skill" is
1 - CRPS_model / CRPS_naive, so positive means the model beats the baseline.

Run from the repository root:  python scripts/backtest_numeric_model.py
It downloads data from FRED and writes numeric_model/BACKTEST.md.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numeric_model import ModelConfig, Series, fetch_fred  # noqa: E402
from numeric_model.simulate import simulate_paths  # noqa: E402

N_PATHS = 2000
YEARS_TESTED = 20  # origins come from roughly the last 20 years
MIN_HISTORY_YEARS = 3  # each origin needs at least this much history


@dataclass(frozen=True)
class SeriesCase:
    series_id: str
    name: str
    variants: tuple[tuple[str, ModelConfig], ...]
    horizons: tuple[int, ...]  # in observations: business days, or weeks for weekly data
    test_max: bool = False  # also test "highest value in the next N observations"


RANDOM_WALK = ModelConfig()
CASES = (
    SeriesCase("SP500", "S&P 500", (("random walk", RANDOM_WALK),), (5, 20, 60), test_max=True),
    SeriesCase("NASDAQ100", "Nasdaq-100", (("random walk", RANDOM_WALK),), (5, 20, 60)),
    SeriesCase(
        "VIXCLS",
        "VIX",
        (("random walk", RANDOM_WALK), ("mean-reverting", ModelConfig(mean_reverting=True))),
        (5, 20, 60),
        test_max=True,
    ),
    SeriesCase("DGS10", "10-year Treasury yield", (("random walk", ModelConfig(transform="level")),), (5, 20, 60)),
    SeriesCase("DCOILBRENTEU", "Brent oil", (("random walk", RANDOM_WALK),), (5, 20, 60), test_max=True),
    SeriesCase("DEXUSEU", "US dollars per euro", (("random walk", RANDOM_WALK),), (5, 20, 60)),
    SeriesCase(
        "GASREGW",
        "US gasoline price (weekly)",
        (("random walk", RANDOM_WALK), ("momentum", ModelConfig(momentum=True))),
        (2, 4, 8),
    ),
    SeriesCase(
        "GASDESW",
        "US diesel price (weekly)",
        (("random walk", RANDOM_WALK), ("momentum", ModelConfig(momentum=True))),
        (2, 4, 8),
    ),
)


def crps(samples: np.ndarray, outcome: float) -> float:
    """Sample estimate of CRPS = E|X - y| - 0.5 E|X - X'|."""
    x = np.sort(samples)
    n = len(x)
    term1 = np.mean(np.abs(x - outcome))
    weights = 2 * np.arange(1, n + 1) - n - 1
    term2 = 2.0 * np.sum(weights * x) / (n * n)
    return float(term1 - 0.5 * term2)


def naive_paths(past: Series, steps: int, config: ModelConfig, rng: np.random.Generator) -> np.ndarray:
    """Normal random walk, zero drift, constant volatility from the same history window."""
    history = past
    if config.history_years:
        history = past.since(past.last_date - timedelta(days=int(365.25 * config.history_years)))
    if config.transform == "log":
        history = history.positive_only()
        x = np.log(history.values)
    else:
        x = history.values
    vol = np.std(np.diff(x))
    shocks = rng.normal(0.0, vol, size=(N_PATHS, steps))
    paths = x[-1] + np.cumsum(shocks, axis=1)
    return np.exp(paths) if config.transform == "log" else paths


@dataclass
class Tally:
    n: int = 0
    in50: int = 0
    in80: int = 0
    in90: int = 0
    below10: int = 0
    above90: int = 0
    crps_model: float = 0.0
    crps_naive: float = 0.0
    naive_in80: int = 0

    def add(self, samples: np.ndarray, naive: np.ndarray, outcome: float) -> None:
        q05, q10, q25, q75, q90, q95 = np.quantile(samples, [0.05, 0.1, 0.25, 0.75, 0.9, 0.95])
        n10, n90 = np.quantile(naive, [0.1, 0.9])
        self.n += 1
        self.in50 += q25 <= outcome <= q75
        self.in80 += q10 <= outcome <= q90
        self.in90 += q05 <= outcome <= q95
        self.below10 += outcome < q10
        self.above90 += outcome > q90
        self.naive_in80 += n10 <= outcome <= n90
        self.crps_model += crps(samples, outcome)
        self.crps_naive += crps(naive, outcome)

    def row(self) -> list[str]:
        skill = 1 - self.crps_model / self.crps_naive if self.crps_naive else float("nan")
        pct = lambda k: f"{100 * k / self.n:.0f}%"  # noqa: E731
        return [
            str(self.n), pct(self.in50), pct(self.in80), pct(self.in90),
            f"{pct(self.below10)} / {pct(self.above90)}", pct(self.naive_in80), f"{skill:+.1%}",
        ]


def origins_for(series: Series, max_horizon: int) -> list[int]:
    weekly = series.median_step_days() > 1.5
    every = 4 if weekly else 21
    first_allowed = np.datetime64(series.dates[0] + np.timedelta64(int(365.25 * MIN_HISTORY_YEARS), "D"))
    earliest = np.datetime64(series.dates[-1] - np.timedelta64(int(365.25 * YEARS_TESTED), "D"))
    start_date = max(first_allowed, earliest)
    start = int(np.searchsorted(series.dates, start_date))
    return list(range(start, len(series.values) - max_horizon, every))


def run_case(case: SeriesCase) -> list[list[str]]:
    full = fetch_fred(case.series_id)
    max_h = max(case.horizons)
    origins = origins_for(full, max_h)
    rows: list[list[str]] = []
    for label, config in case.variants:
        data = full.positive_only() if config.transform == "log" else full
        tallies: dict[tuple[str, int], Tally] = {}
        for origin in origins:
            if origin >= len(data.values) - max_h:
                continue
            past = Series(data.series_id, data.dates[: origin + 1], data.values[: origin + 1])
            rng = np.random.default_rng(origin)
            paths = simulate_paths(past, max_h, config, n_paths=N_PATHS, seed=origin)
            naive = naive_paths(past, max_h, config, rng)
            future = data.values[origin + 1 : origin + 1 + max_h]
            for h in case.horizons:
                tallies.setdefault(("value", h), Tally()).add(paths[:, h - 1], naive[:, h - 1], future[h - 1])
                if case.test_max:
                    tallies.setdefault(("max", h), Tally()).add(
                        paths[:, :h].max(axis=1), naive[:, :h].max(axis=1), future[:h].max()
                    )
        unit = "weeks" if full.median_step_days() > 1.5 else "days"
        for (quantity, h), tally in sorted(tallies.items()):
            what = "value" if quantity == "value" else "highest value"
            rows.append([case.name, label, f"{what} in {h} {unit}"] + tally.row())
    return rows


def main() -> None:
    started = time.time()
    header = [
        "Series", "Model", "Forecast", "Forecasts", "In 50% range", "In 80% range",
        "In 90% range", "Below 10th / above 90th", "Naive: in 80% range", "Skill vs naive",
    ]
    rows: list[list[str]] = []
    for case in CASES:
        print(f"Testing {case.name} ({case.series_id})...", flush=True)
        rows += run_case(case)
    table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    table += ["| " + " | ".join(row) + " |" for row in rows]
    run_time = time.time() - started
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report = f"""# Numeric model: historical calibration test

Generated by `scripts/backtest_numeric_model.py` on {stamp} from FRED data
({run_time:.0f}s, {N_PATHS} simulated paths per forecast).

From many past dates (about monthly over the last {YEARS_TESTED} years, or as far
back as each series allows with {MIN_HISTORY_YEARS} years of history first), each
series is forecast using only the data available on that date, then compared
with what actually happened. "Days" are trading days.

How to read it:
- **In 50/80/90% range:** how often the outcome fell inside that central range.
  For a well-calibrated model these should be close to 50%, 80% and 90%.
- **Below 10th / above 90th:** how often the outcome fell below the 10th or above
  the 90th percentile; each should be about 10%. A lopsided pair means the
  forecasts are biased in one direction.
- **Naive: in 80% range:** the same check for a normal random walk with constant
  volatility, for comparison.
- **Skill vs naive:** 1 - CRPS(model) / CRPS(naive). CRPS is a proper score for
  whole forecast distributions; positive skill means the model beats the naive
  baseline.

{chr(10).join(table)}

Not tested: the ICE BofA high-yield credit spread (BAMLH0A0HYM2). FRED only
publishes its last 3 years, which is too little history for this test.
"""
    out = Path(__file__).resolve().parents[1] / "numeric_model" / "BACKTEST.md"
    out.write_text(report)
    print("\n".join(table))
    print(f"\nWrote {out} in {run_time:.0f}s")


if __name__ == "__main__":
    main()
