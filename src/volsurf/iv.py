"""Implied volatility from European option prices.

The solver works on the out-of-the-money option in normalised form. With ``x = ln(F / K)`` and the
total standard deviation ``s = vol * sqrt(t)``, the call price divided by ``sqrt(F * K)`` is

    b(x, s) = exp(x / 2) * N(x / s + s / 2) - exp(-x / 2) * N(x / s - s / 2)

and it increases from 0 to ``exp(x / 2)`` as ``s`` grows (for ``x <= 0``). In-the-money prices are
first turned into the out-of-the-money price through put-call parity, which avoids losing the time
value to rounding. A safeguarded Newton iteration (bisection whenever a Newton step leaves the
current bracket) finds ``s``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .bs import SQRT_2PI


def _normalised_otm_price(x, s):
    """Normalised price of the out-of-the-money option for ``x <= 0``."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        d1 = x / s + 0.5 * s
        d2 = x / s - 0.5 * s
        return np.exp(0.5 * x) * ndtr(d1) - np.exp(-0.5 * x) * ndtr(d2)


def _normalised_vega(x, s):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.exp(-0.5 * (np.square(x / s) + 0.25 * np.square(s))) / SQRT_2PI


def _solve_total_std(x, b, tol, max_iter):
    """Root of ``b(x, s) = b`` in ``s``; Newton on ``log b`` inside a shrinking bracket."""
    lo = np.zeros_like(b)
    hi = np.ones_like(b)
    for _ in range(80):
        short = _normalised_otm_price(x, hi) < b
        if not short.any():
            break
        hi = np.where(short, 2.0 * hi, hi)
    s = np.clip(SQRT_2PI * b, 1e-8, hi)
    log_target = np.log(b)
    for _ in range(max_iter):
        value = _normalised_otm_price(x, s)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f = np.log(value) - log_target
            candidate = s - f * value / _normalised_vega(x, s)
        converged = np.abs(f) <= tol
        if converged.all():
            break
        above = f > 0
        hi = np.where(above, s, hi)
        lo = np.where(above | converged, lo, s)
        inside = np.isfinite(candidate) & (candidate > lo) & (candidate < hi)
        s = np.where(converged, s, np.where(inside, candidate, 0.5 * (lo + hi)))
    return s


def implied_vol(price, spot, strike, t, r=0.0, q=0.0, call=True, tol=1e-13, max_iter=200):
    """Black-Scholes implied volatility, NaN where no volatility reproduces the price.

    That is the case when the price is at or below the intrinsic value, at or above the forward
    (call) or strike (put) bound, or when ``t`` is not positive.
    """
    price, spot, strike, t, r, q = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (price, spot, strike, t, r, q))
    )
    call = np.broadcast_to(np.asarray(call, dtype=bool), price.shape)
    df = np.exp(-r * t)
    fwd = spot * np.exp((r - q) * t)
    undiscounted = price / df

    # Price of the out-of-the-money option: used as is when it is the one that was quoted,
    # otherwise derived through put-call parity from the in-the-money quote.
    x = np.log(fwd / strike)
    quoted_is_otm = np.where(call, x <= 0, x >= 0)
    parity_shift = np.where(call, fwd - strike, strike - fwd)
    otm = np.where(quoted_is_otm, undiscounted, undiscounted - parity_shift)
    b = otm / np.sqrt(fwd * strike)
    x_otm = -np.abs(x)
    valid = np.isfinite(b) & (t > 0) & (b > 0) & (b < np.exp(0.5 * x_otm))

    s = _solve_total_std(
        np.where(valid, x_otm, 0.0), np.where(valid, b, 0.1), tol=tol, max_iter=max_iter
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = s / np.sqrt(t)
    return np.where(valid, vol, np.nan)
