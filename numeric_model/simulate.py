"""
Statistical forecasts for questions about a published data series.

Given a series' history, the model simulates thousands of possible future
paths and reads off the distribution of whatever the question asks about:
the value on a date, the highest or lowest value in a window, or the change
since a reference date.

Method: filtered historical simulation (Barone-Adesi, Giannopoulos and
Vosper, 1999):
1. Transform the series: logs for prices, plain levels for rates and spreads.
2. Optionally take out AR(1) mean reversion, for series such as the VIX that
   drift back towards a long-run level. Otherwise treat it as a random walk
   with no drift.
3. Filter the one-step shocks with a GARCH(1,1) variance recursion using
   standard parameters (alpha 0.06, beta 0.92) and the sample's long-run
   variance. Dividing by that volatility gives standardised shocks that keep
   the series' own fat tails.
4. Simulate by resampling those shocks and scaling them by a volatility that
   starts at today's level and decays back towards the long-run level, so
   calm and turbulent markets get different spreads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np

from numeric_model.data_sources import Series

Quantity = Literal["value", "max", "min", "change", "pct_change"]

DEFAULT_PERCENTILES: tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99,
)


@dataclass(frozen=True)
class ModelConfig:
    """How to model a series."""

    transform: Literal["log", "level"] = "log"
    # AR(1) on levels: for series that drift back to a long-run level (the VIX).
    mean_reverting: bool = False
    # AR(1) on changes: for series whose moves persist for weeks (retail fuel
    # prices, which follow wholesale prices with a lag).
    momentum: bool = False
    history_years: float | None = 10.0  # None uses all available history
    garch_alpha: float = 0.06
    garch_beta: float = 0.92


@dataclass(frozen=True)
class QuestionSpec:
    """What the question asks about the series."""

    end_date: date
    quantity: Quantity = "value"
    # For "max"/"min": the first date that counts. None means since the start
    # of the data, i.e. an all-time high or low.
    window_start: date | None = None
    # For "change"/"pct_change": the date to compare against. None means the
    # latest observation.
    reference_date: date | None = None


@dataclass
class ForecastResult:
    series_id: str
    spec: QuestionSpec
    percentiles: dict[float, float]
    samples: np.ndarray
    last_date: date
    last_value: float
    steps: int
    current_vol: float
    long_run_vol: float
    notes: list[str] = field(default_factory=list)

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    def summary(self) -> str:
        """A short plain-text description, e.g. to show the forecasting model."""
        p = self.percentiles
        vol_ratio = self.current_vol / self.long_run_vol if self.long_run_vol else 1.0
        lines = [
            f"Statistical model for FRED series {self.series_id} "
            f"({self.spec.quantity} by {self.spec.end_date.isoformat()}), "
            f"simulated from its history. Latest observation: {_fmt(self.last_value)} "
            f"on {self.last_date.isoformat()} ({self.steps} future observations to go).",
            f"Median {_fmt(p[0.5])}; 80% range {_fmt(p[0.1])} to {_fmt(p[0.9])}; "
            f"90% range {_fmt(p[0.05])} to {_fmt(p[0.95])}.",
            f"Current volatility is {vol_ratio:.2f}x its long-run average.",
        ]
        return "\n".join(lines + self.notes)


def _fmt(value: float) -> str:
    return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.0f}"


def steps_between(series: Series, start: date, end: date) -> int:
    """
    How many observations the series will publish after `start`, up to and
    including `end`: business days for daily data, otherwise the number of
    typical gaps (e.g. weeks for weekly data).
    """
    if end <= start:
        return 0
    if series.median_step_days() <= 1.5:
        return int(
            np.busday_count(
                np.datetime64(start, "D") + np.timedelta64(1, "D"),
                np.datetime64(end, "D") + np.timedelta64(1, "D"),
            )
        )
    return int(round((end - start).days / series.median_step_days()))


@dataclass(frozen=True)
class _FittedModel:
    x_last: float
    intercept: float
    slope: float  # 1.0 for a random walk
    shocks_std: np.ndarray  # standardised shocks to resample (mean 0, std 1)
    next_var: float
    long_run_var: float
    alpha: float
    beta: float
    momentum: float = 0.0  # AR(1) coefficient on changes; 0 means none
    last_change: float = 0.0


def _fit(series: Series, config: ModelConfig) -> _FittedModel:
    history = series
    if config.history_years:
        start = series.last_date - timedelta(days=int(365.25 * config.history_years))
        history = series.since(start)
    if config.transform == "log":
        history = history.positive_only()
        x = np.log(history.values)
    else:
        x = history.values.astype(float)
    if len(x) < 60:
        raise ValueError(
            f"Only {len(x)} usable observations of {series.series_id}; need at least 60"
        )

    if config.mean_reverting and config.momentum:
        raise ValueError("Choose either mean_reverting or momentum, not both")
    momentum, last_change = 0.0, 0.0
    if config.mean_reverting:
        slope, intercept = np.polyfit(x[:-1], x[1:], 1)
        slope = float(min(slope, 0.9999))  # keep the process stationary
        intercept = float(intercept)
        shocks = x[1:] - (intercept + slope * x[:-1])
    elif config.momentum:
        slope, intercept = 1.0, 0.0
        changes = np.diff(x)
        changes = changes - changes.mean()  # no drift
        momentum = float(np.dot(changes[:-1], changes[1:]) / np.dot(changes[:-1], changes[:-1]))
        momentum = max(min(momentum, 0.95), -0.95)
        shocks = changes[1:] - momentum * changes[:-1]
        last_change = float(changes[-1])
    else:
        slope, intercept = 1.0, 0.0
        shocks = np.diff(x)
    shocks = shocks - shocks.mean()  # no drift

    alpha, beta = config.garch_alpha, config.garch_beta
    long_run_var = float(np.var(shocks))
    if long_run_var <= 0:
        raise ValueError(f"{series.series_id} has no variation to model")
    omega = long_run_var * (1 - alpha - beta)
    var = long_run_var
    sigmas = np.empty(len(shocks))
    for t, shock in enumerate(shocks):
        sigmas[t] = np.sqrt(var)
        var = omega + alpha * shock * shock + beta * var
    standardised = shocks / sigmas
    standardised = (standardised - standardised.mean()) / standardised.std()
    return _FittedModel(
        x_last=float(x[-1]),
        intercept=intercept,
        slope=slope,
        shocks_std=standardised,
        next_var=float(var),
        long_run_var=long_run_var,
        alpha=alpha,
        beta=beta,
        momentum=momentum,
        last_change=last_change,
    )


def _simulate(
    model: _FittedModel, steps: int, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulated transformed values, shape (n_paths, steps)."""
    omega = model.long_run_var * (1 - model.alpha - model.beta)
    paths = np.empty((n_paths, steps))
    x = np.full(n_paths, model.x_last)
    var = np.full(n_paths, model.next_var)
    previous_change = np.full(n_paths, model.last_change)
    for step in range(steps):
        shock = np.sqrt(var) * rng.choice(model.shocks_std, size=n_paths)
        if model.momentum:
            change = model.momentum * previous_change + shock
            x = x + change
            previous_change = change
        else:
            x = model.intercept + model.slope * x + shock
        paths[:, step] = x
        var = omega + model.alpha * shock * shock + model.beta * var
    return paths


def simulate_paths(
    series: Series,
    steps: int,
    config: ModelConfig = ModelConfig(),
    n_paths: int = 5000,
    seed: int | None = 0,
) -> np.ndarray:
    """Simulated future values in the series' own units, shape (n_paths, steps)."""
    model = _fit(series, config)
    simulated = _simulate(model, steps, n_paths, np.random.default_rng(seed))
    return np.exp(simulated) if config.transform == "log" else simulated


def _value_on_or_before(series: Series, day: date) -> float:
    index = int(np.searchsorted(series.dates, np.datetime64(day, "D"), side="right")) - 1
    if index < 0:
        raise ValueError(f"{series.series_id} has no data on or before {day}")
    return float(series.values[index])


def forecast(
    series: Series,
    spec: QuestionSpec,
    config: ModelConfig = ModelConfig(),
    n_paths: int = 5000,
    seed: int | None = 0,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> ForecastResult:
    """Simulates the distribution of `spec.quantity` for `series`."""
    last_date = series.last_date
    steps = steps_between(series, last_date, spec.end_date)
    model = _fit(series, config)
    notes: list[str] = []

    if steps <= 0:
        # No new observations will arrive before the end date, so the answer
        # is (almost) already known from the data.
        notes.append("No new observations are expected before the end date.")
        paths = np.full((n_paths, 1), series.last_value)
    else:
        rng = np.random.default_rng(seed)
        simulated = _simulate(model, steps, n_paths, rng)
        paths = np.exp(simulated) if config.transform == "log" else simulated

    end_values = paths[:, -1]
    if spec.quantity == "value":
        samples = end_values
    elif spec.quantity in ("max", "min"):
        pick = np.max if spec.quantity == "max" else np.min
        combine = np.maximum if spec.quantity == "max" else np.minimum
        first_step = 0
        observed_extreme: float | None = None
        if spec.window_start is None:
            # All-time high or low: the history counts too.
            observed_extreme = float(pick(series.values))
        elif spec.window_start <= last_date:
            # The window has already started: what's been observed so far counts.
            in_window = series.values[series.dates >= np.datetime64(spec.window_start, "D")]
            if len(in_window) > 0:
                observed_extreme = float(pick(in_window))
        else:
            first_step = steps_between(
                series, last_date, spec.window_start - timedelta(days=1)
            )
            if first_step >= paths.shape[1]:
                raise ValueError("The window starts after the end date")
        samples = pick(paths[:, first_step:], axis=1)
        if observed_extreme is not None:
            samples = combine(samples, observed_extreme)
    elif spec.quantity in ("change", "pct_change"):
        reference = spec.reference_date or last_date
        if reference <= last_date:
            reference_values = np.full(n_paths, _value_on_or_before(series, reference))
        else:
            ref_step = steps_between(series, last_date, reference)
            reference_values = paths[:, max(ref_step - 1, 0)]
        if spec.quantity == "change":
            samples = end_values - reference_values
        else:
            samples = (end_values / reference_values - 1.0) * 100.0
    else:
        raise ValueError(f"Unknown quantity: {spec.quantity}")

    return ForecastResult(
        series_id=series.series_id,
        spec=spec,
        percentiles={q: float(v) for q, v in zip(percentiles, np.quantile(samples, percentiles))},
        samples=samples,
        last_date=last_date,
        last_value=series.last_value,
        steps=steps,
        current_vol=float(np.sqrt(model.next_var)),
        long_run_vol=float(np.sqrt(model.long_run_var)),
        notes=notes,
    )
