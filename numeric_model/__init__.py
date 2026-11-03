"""
Data-driven forecasts for numeric questions about published data series
(market prices, yields, fuel prices, economic indicators).

See simulate.py for the method and BACKTEST.md for how well calibrated it is
on historical data.
"""

from numeric_model.data_sources import Series, fetch_fred, parse_fred_csv
from numeric_model.registry import KNOWN_SERIES, KnownSeries, config_for
from numeric_model.simulate import (
    DEFAULT_PERCENTILES,
    ForecastResult,
    ModelConfig,
    QuestionSpec,
    forecast,
    steps_between,
)

__all__ = [
    "DEFAULT_PERCENTILES",
    "ForecastResult",
    "KNOWN_SERIES",
    "KnownSeries",
    "ModelConfig",
    "QuestionSpec",
    "Series",
    "config_for",
    "fetch_fred",
    "forecast",
    "parse_fred_csv",
    "steps_between",
]
