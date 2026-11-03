"""
Blends the statistical model's distribution with the AI's forecast.

The blend is a mixture of the two cumulative distribution functions (CDFs),
evaluated on the question's own grid: the 201 points (fewer for discrete
questions) at which Metaculus stores a forecast. Working on that grid keeps
the question's bounds and its discrete outcomes exactly as Metaculus sees
them.

Metaculus rejects CDFs that put no mass beyond an open bound, that are flat
anywhere, or that put too much mass in one bucket. The AI's CDF from
forecasting-tools already meets those rules. The model's CDF is made to meet
them the same way (see standardize_model_cdf), so any mixture of the two does
too.

No dependency on forecasting-tools: main.py passes plain lists in and turns
the result back into Percentile objects.
"""

from __future__ import annotations

import numpy as np

# Metaculus's limits (from its API docs, as implemented in forecasting-tools):
# the most mass one bucket may hold is 0.2 for the standard 200 buckets, and
# consecutive CDF values must differ by at least 5e-5.
_MAX_PMF_FOR_200_BUCKETS = 0.2
MIN_CDF_STEP = 5e-5


def max_pmf(cdf_size: int) -> float:
    return _MAX_PMF_FOR_200_BUCKETS * 200 / (cdf_size - 1)


def model_cdf_on_grid(
    samples: np.ndarray,
    grid_values: np.ndarray,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> np.ndarray:
    """
    The model's CDF at each grid value. As on Metaculus, the first point is
    P(outcome < lower bound) and every other point is P(outcome <= value).
    Outcomes beyond a closed bound can't happen, so samples there count as
    landing on the bound.
    """
    grid_values = np.asarray(grid_values, dtype=float)
    s = np.sort(np.asarray(samples, dtype=float))
    lower, upper = grid_values[0], grid_values[-1]
    if not open_lower_bound:
        s = np.maximum(s, lower)
    if not open_upper_bound:
        s = np.minimum(s, upper)
    n = len(s)
    cdf = np.searchsorted(s, grid_values, side="right") / n
    cdf[0] = np.searchsorted(s, lower, side="left") / n
    return cdf


def standardize_model_cdf(
    cdf: np.ndarray, open_lower_bound: bool, open_upper_bound: bool
) -> np.ndarray:
    """
    Applies Metaculus's rules to the model's CDF, using the same formula as
    forecasting-tools: about 1% of the mass is spread evenly across the range (so the
    CDF is never flat), and 0.1% goes beyond each open bound. If one bucket
    still holds more than 90% of the allowed maximum (the model is nearly
    certain), more mass is spread evenly until it doesn't.
    """
    cdf = np.asarray(cdf, dtype=float)
    size = len(cdf)
    location = np.linspace(0.0, 1.0, size)
    low_to = 0.0 if open_lower_bound else cdf[0]
    high_to = 1.0 if open_upper_bound else cdf[-1]
    inbound = high_to - low_to
    rescaled = (cdf - low_to) / inbound if inbound > 0 else location.copy()
    # forecasting-tools spreads 1% evenly, which makes the CDF rise by exactly
    # the minimum step where it would otherwise be flat. 1.1% here keeps every
    # step of a blend strictly above the minimum despite rounding.
    spread = 0.011
    if open_lower_bound and open_upper_bound:
        out = (0.998 - spread) * rescaled + spread * location + 0.001
    elif open_lower_bound:
        out = (0.999 - spread) * rescaled + spread * location + 0.001
    elif open_upper_bound:
        out = (0.999 - spread) * rescaled + spread * location
    else:
        out = (1.0 - spread) * rescaled + spread * location

    cap = 0.9 * max_pmf(size)
    pmf = np.diff(out)
    if pmf.max() > cap:
        # Mix with a uniform spread over the range: (1 - u) * out + u * even.
        start, end = out[0], out[-1]
        even = start + (end - start) * location
        even_pmf = (end - start) / (size - 1)
        u = (pmf.max() - cap) / (pmf.max() - even_pmf)
        out = (1 - u) * out + u * even
    return out


def blend_cdfs(
    ai_cdf: np.ndarray,
    grid_values: np.ndarray,
    model_samples: np.ndarray,
    weight: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> np.ndarray:
    """
    The mixture weight * model + (1 - weight) * AI, on the AI CDF's grid.
    `ai_cdf` must be a CDF that already meets Metaculus's rules (e.g. the
    output of forecasting-tools' NumericDistribution.get_cdf()).
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight}")
    ai_cdf = np.asarray(ai_cdf, dtype=float)
    if ai_cdf.shape != np.shape(grid_values):
        raise ValueError("The AI CDF and the grid have different lengths")
    model = standardize_model_cdf(
        model_cdf_on_grid(model_samples, grid_values, open_lower_bound, open_upper_bound),
        open_lower_bound,
        open_upper_bound,
    )
    blended = weight * model + (1.0 - weight) * ai_cdf
    check_cdf(blended, open_lower_bound, open_upper_bound)
    return blended


def check_cdf(cdf: np.ndarray, open_lower_bound: bool, open_upper_bound: bool) -> None:
    """Raises ValueError if the CDF breaks one of Metaculus's rules."""
    cdf = np.asarray(cdf, dtype=float)
    if not np.all(np.isfinite(cdf)):
        raise ValueError("The CDF has non-finite values")
    steps = np.diff(cdf)
    if steps.min() < MIN_CDF_STEP * (1 - 1e-6):
        raise ValueError(f"The CDF rises by only {steps.min():.2g} somewhere")
    if steps.max() > max_pmf(len(cdf)):
        raise ValueError(f"One bucket holds {steps.max():.3f} of the probability")
    if open_lower_bound and cdf[0] <= 0:
        raise ValueError("No probability below an open lower bound")
    if not open_lower_bound and abs(cdf[0]) > 1e-9:
        raise ValueError("Probability below a closed lower bound")
    if open_upper_bound and cdf[-1] >= 1:
        raise ValueError("No probability above an open upper bound")
    if not open_upper_bound and abs(cdf[-1] - 1) > 1e-9:
        raise ValueError("Probability above a closed upper bound")
