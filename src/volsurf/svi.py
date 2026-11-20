"""Raw SVI parametrisation of one expiry slice, its fit and its static no-arbitrage conditions.

With ``k = ln(K / F)`` the log-moneyness and ``w = vol^2 * t`` the total implied variance, the raw
SVI slice of Gatheral is

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)).

The fit uses the reduction of Zeliade Systems: for fixed ``(m, sigma)`` the model is linear in the
other three parameters, so only a two-dimensional search is left, which is then polished with a
bounded least-squares step on all five parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize


@dataclass(frozen=True)
class SviSlice:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def slope(self, k):
        """dw/dk."""
        x = np.asarray(k, dtype=float) - self.m
        return self.b * (self.rho + x / np.sqrt(x**2 + self.sigma**2))

    def curvature(self, k):
        """d2w/dk2."""
        x = np.asarray(k, dtype=float) - self.m
        return self.b * self.sigma**2 / (x**2 + self.sigma**2) ** 1.5

    @property
    def minimum_total_variance(self) -> float:
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    def is_admissible(self, tol: float = 1e-12) -> bool:
        """Parameter domain on which ``w`` is a non-negative total variance (up to rounding)."""
        return bool(
            self.b >= 0
            and abs(self.rho) < 1
            and self.sigma > 0
            and self.minimum_total_variance >= -tol
        )

    def satisfies_lee_bound(self) -> bool:
        """Wing slopes of ``w`` must not exceed 2 (Lee's moment formula)."""
        return bool(self.b * (1 + self.rho) <= 2 and self.b * (1 - self.rho) <= 2)

    def butterfly_g(self, k):
        """Gatheral-Jacquier ``g``; the slice has no butterfly arbitrage iff ``g >= 0``."""
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        w1 = self.slope(k)
        w2 = self.curvature(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1**2 * (1.0 / w + 0.25) + 0.5 * w2

    def density(self, k):
        """Risk-neutral density of ``ln(S_T / F)`` implied by the slice."""
        k = np.asarray(k, dtype=float)
        w = self.total_variance(k)
        with np.errstate(divide="ignore", invalid="ignore"):
            d2 = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
            return self.butterfly_g(k) * np.exp(-0.5 * d2**2) / np.sqrt(2.0 * np.pi * w)


@dataclass(frozen=True)
class SviFit:
    slice: SviSlice
    rmse: float
    max_abs_error: float


def _linear_core(k, w, weights, m, sigma):
    """Least squares in (a, d, c) for fixed (m, sigma), projected onto the admissible domain."""
    y = (k - m) / sigma
    design = np.column_stack([np.ones_like(y), y, np.sqrt(y**2 + 1.0)])
    sqrt_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], w * sqrt_w, rcond=None)
    a, d, c = coef
    c = max(c, 1e-10)
    d = float(np.clip(d, -0.999 * c, 0.999 * c))
    a = max(a, -np.sqrt(c**2 - d**2))
    residual = (w - (a + d * y + c * design[:, 2])) * sqrt_w
    return a, d, c, float(residual @ residual)


def _to_slice(a, d, c, m, sigma):
    return SviSlice(a=a, b=c / sigma, rho=d / c, m=m, sigma=sigma)


def _lift_to_nonnegative(s: SviSlice) -> SviSlice:
    """Raise ``a`` just enough that the smallest total variance is not below zero.

    The polishing step only penalises a negative minimum, so it may stop a hair under zero.
    """
    floor = -s.b * s.sigma * np.sqrt(1.0 - s.rho**2)
    return s if s.a >= floor else SviSlice(floor, s.b, s.rho, s.m, s.sigma)


def fit_slice(k, w, weights=None, n_starts=6, seed=0) -> SviFit:
    """Fit a raw SVI slice to total variances ``w`` observed at log-moneyness ``k``."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if k.shape != w.shape or k.ndim != 1 or len(k) < 5:
        raise ValueError("need at least five (k, w) pairs of the same length")
    weights = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    span = max(k.max() - k.min(), 1e-3)
    m_low, m_high = k.min() - span, k.max() + span
    sigma_low, sigma_high = max(1e-4 * span, 1e-5), 2.0 * span

    def inside_box(theta):
        """Search variables (m, ln sigma) as parameters, clipped to where the fit may live.

        A nearly straight smile is matched equally well by a vertex far away with a large sigma, so
        without this the search can run off to an arbitrarily distant vertex.
        """
        m = float(np.clip(theta[0], m_low, m_high))
        return m, float(np.clip(np.exp(theta[1]), sigma_low, sigma_high))

    def objective(theta):
        m, sigma = inside_box(theta)
        return _linear_core(k, w, weights, m, sigma)[3]

    rng = np.random.default_rng(seed)
    k_at_min = k[np.argmin(w)]
    starts = [(k_at_min, np.log(0.1 * span)), (0.0, np.log(0.3 * span))]
    # Corners of the search box: a nearly straight smile is fitted by a vertex outside the quotes.
    starts += [
        (m_low, np.log(0.05 * span)),
        (m_high, np.log(0.05 * span)),
        (k_at_min, np.log(3.0 * span)),
    ]
    starts += [
        (k_at_min + rng.normal(0, 0.2 * span), np.log(span * rng.uniform(0.02, 0.8)))
        for _ in range(max(n_starts - 2, 0))
    ]
    best = None
    for start in starts:
        result = minimize(
            objective, start, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-16}
        )
        if best is None or result.fun < best.fun:
            best = result
    m, sigma = inside_box(best.x)
    a, d, c, _ = _linear_core(k, w, weights, m, sigma)
    start = SviSlice(a=a, b=c / sigma, rho=d / c, m=m, sigma=sigma)

    sqrt_w = np.sqrt(weights)

    def residuals(theta):
        trial = SviSlice(*theta)
        penalty = 1e3 * max(-trial.minimum_total_variance, 0.0)
        return np.append((trial.total_variance(k) - w) * sqrt_w, penalty)

    # rho = d / c can exceed 0.999 by one unit in the last place after the clipping in _linear_core
    lower = np.array([-np.inf, 0.0, -0.999, m_low, 1e-6])
    upper = np.array([np.inf, np.inf, 0.999, m_high, np.inf])
    theta0 = np.clip([start.a, start.b, start.rho, start.m, start.sigma], lower, upper)
    polished = least_squares(residuals, theta0, bounds=(lower, upper), xtol=1e-14, ftol=1e-14)
    final = SviSlice(*polished.x)
    if np.sum(residuals(polished.x) ** 2) > np.sum(residuals(theta0) ** 2):
        final = start
    final = _lift_to_nonnegative(final)
    error = final.total_variance(k) - w
    return SviFit(final, float(np.sqrt(np.mean(error**2))), float(np.max(np.abs(error))))


def butterfly_violations(slice_: SviSlice, k_grid) -> np.ndarray:
    """Grid points where the implied density would be negative."""
    k_grid = np.asarray(k_grid, dtype=float)
    return k_grid[slice_.butterfly_g(k_grid) < -1e-10]
