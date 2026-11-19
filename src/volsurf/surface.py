"""A volatility surface made of SVI slices, with calendar-spread repair and time interpolation.

Slices are fitted expiry by expiry. A fitted slice must not fall below the previous one at any
log-moneyness (otherwise a calendar spread would have negative value); when it does, the offending
quotes are raised to the previous slice and the slice is fitted again. Between expiries the total
variance is interpolated linearly in time at fixed log-moneyness, which keeps the surface free of
calendar arbitrage whenever the slices are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svi import SviSlice, butterfly_violations, fit_slice


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


def fit_surface(quotes, k_grid=None, repair_calendar=True, max_repairs=8, **fit_kwargs):
    """Fit a surface to quotes sorted or unsorted by expiry; returns the surface and the fits."""
    quotes = sorted(quotes, key=lambda q: q.t)
    if k_grid is None:
        low = min(q.k.min() for q in quotes)
        high = max(q.k.max() for q in quotes)
        k_grid = np.linspace(low - 0.2, high + 0.2, 401)
    fits = []
    for quote in quotes:
        w = np.asarray(quote.w, dtype=float).copy()
        fit = fit_slice(quote.k, w, **fit_kwargs)
        if repair_calendar and fits:
            previous = fits[-1].slice
            for _ in range(max_repairs):
                gap = fit.slice.total_variance(k_grid) - previous.total_variance(k_grid)
                if gap.min() >= -1e-10:
                    break
                floor = previous.total_variance(quote.k) + 1e-6
                w = np.maximum(w, floor)
                fit = fit_slice(quote.k, w, **fit_kwargs)
        fits.append(fit)
    surface = Surface([(q.t, f.slice) for q, f in zip(quotes, fits, strict=True)])
    return surface, fits
