"""
Downloads public data series for the numeric model.

FRED (https://fred.stlouisfed.org), run by the St. Louis Fed, publishes
thousands of economic and market series as plain CSV files with no API key,
e.g. SP500 (S&P 500), VIXCLS (VIX), DGS10 (10-year Treasury yield) and
GASREGW (US regular gasoline price).
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import requests

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_USER_AGENT = "metac-bot-numeric-model"


@dataclass(frozen=True)
class Series:
    """Dated observations, oldest first, with missing values removed."""

    series_id: str
    dates: np.ndarray  # numpy datetime64[D]
    values: np.ndarray  # float64

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values must have the same length")
        if len(self.dates) == 0:
            raise ValueError(f"Series {self.series_id} has no observations")

    @property
    def last_date(self) -> date:
        return self.dates[-1].astype(object)

    @property
    def last_value(self) -> float:
        return float(self.values[-1])

    def median_step_days(self) -> float:
        """Typical gap between observations: about 1 for daily data, 7 for weekly."""
        if len(self.dates) < 2:
            return 1.0
        gaps = np.diff(self.dates).astype("timedelta64[D]").astype(float)
        return float(np.median(gaps))

    def up_to(self, cutoff: date) -> Series:
        """Only the observations on or before `cutoff` (used by backtests)."""
        keep = self.dates <= np.datetime64(cutoff, "D")
        return Series(self.series_id, self.dates[keep], self.values[keep])

    def since(self, start: date) -> Series:
        """Only the observations on or after `start`."""
        keep = self.dates >= np.datetime64(start, "D")
        return Series(self.series_id, self.dates[keep], self.values[keep])

    def positive_only(self) -> Series:
        """Drops zero and negative values, which a log transform can't handle
        (e.g. the one day in April 2020 when WTI oil closed below zero)."""
        keep = self.values > 0
        return Series(self.series_id, self.dates[keep], self.values[keep])


def parse_fred_csv(series_id: str, text: str) -> Series:
    """
    Parses FRED's CSV format: a header row (date column, then the series ID)
    followed by one row per date. Missing values appear as "." or blanks.
    """
    reader = csv.reader(io.StringIO(text.strip()))
    header = next(reader, None)
    if not header or len(header) < 2:
        raise ValueError(f"Unexpected FRED CSV header for {series_id}: {header!r}")
    dates: list[np.datetime64] = []
    values: list[float] = []
    for row in reader:
        if len(row) < 2:
            continue
        raw_date, raw_value = row[0].strip(), row[1].strip()
        if raw_value in ("", "."):
            continue
        try:
            value = float(raw_value)
            day = np.datetime64(raw_date, "D")
        except ValueError:
            continue
        dates.append(day)
        values.append(value)
    order = np.argsort(np.array(dates, dtype="datetime64[D]"), kind="stable")
    return Series(
        series_id=series_id,
        dates=np.array(dates, dtype="datetime64[D]")[order],
        values=np.array(values, dtype=float)[order],
    )


_cache: dict[str, Series] = {}


def fetch_fred(series_id: str, timeout: float = 30.0) -> Series:
    """
    Downloads a FRED series (cached for the rest of the run). Note that FRED
    usually lags markets by one or two business days.
    """
    series_id = series_id.strip().upper()
    if series_id in _cache:
        return _cache[series_id]
    response = requests.get(
        FRED_CSV_URL.format(series_id=series_id),
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    series = parse_fred_csv(series_id, response.text)
    logger.info(
        f"Fetched FRED {series_id}: {len(series.values)} observations, "
        f"last {series.last_date} = {series.last_value}"
    )
    _cache[series_id] = series
    return series
