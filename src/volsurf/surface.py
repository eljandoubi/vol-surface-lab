"""A volatility surface made of SVI slices, with calendar-spread repair and time interpolation.

Slices are fitted expiry by expiry. A fitted slice must not fall below the previous one at any
log-moneyness (otherwise a calendar spread would have negative value). When it does, the slice is
fitted again with one more pseudo-quote at the worst violating point, equal to the previous slice
there, and this is repeated until the violation is gone. Raising the real quotes would not do: a
violation between or beyond the quotes does not move when the quotes move. Between expiries the
total variance is interpolated linearly in time at fixed log-moneyness, which keeps the surface
free of calendar arbitrage whenever the slices are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svi import SviFit, SviSlice, butterfly_violations, fit_slice


@dataclass(frozen=True)
class SliceQuotes:
    """Total variances ``w`` observed at log-moneyness ``k`` for one expiry ``t`` (in years)."""

    t: float
    k: np.ndarray
    w: np.ndarray


class Surface:
    def __init__(self, slices):
        slices = sorted(slices, key=lambda item: item[0])
        self.times = np.array([t for t, _ in slices], dtype=float)
        self.slices: list[SviSlice] = [s for _, s in slices]
        if np.any(np.diff(self.times) <= 0) or self.times[0] <= 0:
            raise ValueError("expiries must be positive and distinct")

    def total_variance(self, k, t):
        """Total variance at log-moneyness ``k`` and expiry ``t`` (arrays broadcast)."""
        k = np.asarray(k, dtype=float)
        t = float(t)
        values = np.array([s.total_variance(k) for s in self.slices])
        if t <= self.times[0]:
            return values[0] * t / self.times[0]
        if t >= self.times[-1]:
            return values[-1] * t / self.times[-1]
        j = int(np.searchsorted(self.times, t))
        weight = (t - self.times[j - 1]) / (self.times[j] - self.times[j - 1])
        return (1 - weight) * values[j - 1] + weight * values[j]

    def implied_vol(self, k, t):
        return np.sqrt(self.total_variance(k, t) / t)

    def calendar_violations(self, k_grid, tol=1e-10):
        """List of (earlier expiry, later expiry, worst shortfall) for every violated pair."""
        k_grid = np.asarray(k_grid, dtype=float)
        out = []
        for i in range(len(self.slices) - 1):
            gap = self.slices[i + 1].total_variance(k_grid) - self.slices[i].total_variance(k_grid)
            if gap.min() < -tol:
                out.append((float(self.times[i]), float(self.times[i + 1]), float(gap.min())))
        return out

    def butterfly_violations(self, k_grid):
        """List of (expiry, number of violating grid points)."""
        found = []
        for t, s in zip(self.times, self.slices, strict=True):
            bad = butterfly_violations(s, k_grid)
            if bad.size:
                found.append((float(t), int(bad.size)))
        return found


def _quoted_overlap(k_early, k_late, n=201):
    """Grid over the range both expiries quote, or None when the ranges do not overlap."""
    low, high = max(k_early.min(), k_late.min()), min(k_early.max(), k_late.max())
    return np.linspace(low, high, n) if high > low else None


def fit_surface(quotes, k_grid=None, repair_calendar=True, max_repairs=12, **fit_kwargs):
    """Fit a surface to quotes sorted or unsorted by expiry; returns the surface and the fits.

    Calendar spreads are checked on ``k_grid``; by default on the range quoted for both expiries of
    each consecutive pair, because outside it a slice is an extrapolation that no data constrains.
    A violation is repaired by raising the quotes that lie below the previous slice and by adding
    a heavy pseudo-quote at the worst violating point (a violation between or beyond the quotes
    does not move when the quotes move). The errors stored in each fit are measured against the
    original quotes.
    """
    quotes = sorted(quotes, key=lambda q: q.t)
    fixed_grid = None if k_grid is None else np.asarray(k_grid, dtype=float)
    fits = []
    for i, quote in enumerate(quotes):
        kwargs = dict(fit_kwargs)
        k_real = np.asarray(quote.k, dtype=float)
        n = len(k_real)
        w_fit = np.asarray(quote.w, dtype=float).copy()
        k_fit = k_real.copy()
        weights = np.asarray(kwargs.pop("weights", np.ones(n)), dtype=float)
        fit = fit_slice(k_fit, w_fit, weights=weights, **kwargs)
        grid = fixed_grid
        if repair_calendar and fits:
            if grid is None:
                grid = _quoted_overlap(np.asarray(quotes[i - 1].k), k_real)
            previous = fits[-1].slice
            for step in range(max_repairs if grid is not None else 0):
                gap = fit.slice.total_variance(grid) - previous.total_variance(grid)
                worst = int(np.argmin(gap))
                if gap[worst] >= -1e-10:
                    break
                w_fit[:n] = np.maximum(w_fit[:n], previous.total_variance(k_real) + 1e-6)
                k_fit = np.append(k_fit, grid[worst])
                w_fit = np.append(w_fit, previous.total_variance(grid[worst]) + 1e-6)
                weights = np.append(weights, 10.0 * 2.0**step)
                fit = fit_slice(k_fit, w_fit, weights=weights, **kwargs)
        error = fit.slice.total_variance(k_real) - quote.w
        fits.append(
            SviFit(fit.slice, float(np.sqrt(np.mean(error**2))), float(np.max(np.abs(error))))
        )
    surface = Surface([(q.t, f.slice) for q, f in zip(quotes, fits, strict=True)])
    return surface, fits
