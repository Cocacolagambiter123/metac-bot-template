"""
FRED series the numeric model knows how to handle, with the model that fitted
each one best in the historical test (see BACKTEST.md).

The list doubles as a menu for matching questions to data: a question about
"the 10-year Treasury yield on December 31" maps to DGS10.
"""

from __future__ import annotations

from dataclasses import dataclass

from numeric_model.data_sources import Series
from numeric_model.simulate import ModelConfig


@dataclass(frozen=True)
class KnownSeries:
    series_id: str
    description: str
    units: str
    config: ModelConfig
    tested: bool  # True if covered by the historical test


_RANDOM_WALK = ModelConfig()
_LEVEL = ModelConfig(transform="level")
_MEAN_REVERTING = ModelConfig(mean_reverting=True)
_MOMENTUM = ModelConfig(momentum=True)

KNOWN_SERIES: dict[str, KnownSeries] = {
    s.series_id: s
    for s in (
        KnownSeries("SP500", "S&P 500 index, daily close", "index points", _RANDOM_WALK, True),
        KnownSeries("NASDAQ100", "Nasdaq-100 index, daily close", "index points", _RANDOM_WALK, True),
        KnownSeries("NASDAQCOM", "Nasdaq Composite index, daily close", "index points", _RANDOM_WALK, False),
        KnownSeries("DJIA", "Dow Jones Industrial Average, daily close", "index points", _RANDOM_WALK, False),
        KnownSeries("VIXCLS", "CBOE VIX volatility index, daily close", "index points", _MEAN_REVERTING, True),
        KnownSeries("DGS10", "10-year US Treasury yield, daily", "percent", _LEVEL, True),
        KnownSeries("DGS2", "2-year US Treasury yield, daily", "percent", _LEVEL, False),
        KnownSeries("DGS30", "30-year US Treasury yield, daily", "percent", _LEVEL, False),
        KnownSeries("DCOILBRENTEU", "Brent crude oil spot price, daily", "US dollars per barrel", _RANDOM_WALK, True),
        KnownSeries("DCOILWTICO", "WTI crude oil spot price, daily", "US dollars per barrel", _RANDOM_WALK, False),
        KnownSeries("DEXUSEU", "US dollars per euro, daily", "US dollars", _RANDOM_WALK, True),
        KnownSeries("DEXJPUS", "Japanese yen per US dollar, daily", "yen", _RANDOM_WALK, False),
        KnownSeries("DEXUSUK", "US dollars per British pound, daily", "US dollars", _RANDOM_WALK, False),
        KnownSeries("CBBTCUSD", "Bitcoin price on Coinbase, daily", "US dollars", _RANDOM_WALK, False),
        KnownSeries("GASREGW", "US regular gasoline retail price, weekly (EIA)", "US dollars per gallon", _MOMENTUM, True),
        KnownSeries("GASDESW", "US diesel retail price, weekly (EIA)", "US dollars per gallon", _MOMENTUM, True),
        # FRED only publishes the last 3 years of this index, too little to test.
        KnownSeries("BAMLH0A0HYM2", "ICE BofA US high-yield option-adjusted spread, daily", "percent", _MEAN_REVERTING, False),
    )
}


def config_for(series: Series) -> tuple[ModelConfig, bool]:
    """
    The model to use for a series, and whether that choice was tested. Series
    outside the list get a plain random walk: in logs if every value is
    positive, otherwise in levels.
    """
    known = KNOWN_SERIES.get(series.series_id)
    if known is not None:
        return known.config, known.tested
    if (series.values > 0).all():
        return _RANDOM_WALK, False
    return _LEVEL, False
