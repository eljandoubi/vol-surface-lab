"""Heston stochastic volatility model.

    dS = (r - q) S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + xi sqrt(v) dW2,     corr(dW1, dW2) = rho

Prices come from Lewis' single-integral formula with the characteristic function written in the
"little Heston trap" form, which stays continuous across the branch cut of the complex logarithm.
Paths come from Andersen's quadratic-exponential (QE) scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .iv import implied_vol


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        for name in ("v0", "kappa", "theta", "xi"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")

    @property
    def feller(self) -> float:
        """Positive when the variance process cannot reach zero."""
        return 2.0 * self.kappa * self.theta - self.xi**2

    def with_v0(self, v0: float) -> HestonParams:
        return HestonParams(v0, self.kappa, self.theta, self.xi, self.rho)


def _exponent_terms(u, t, p: HestonParams):
    """The functions C(u, t) and D(u, t) with psi(u) = exp(C + D * v0)."""
    u = np.asarray(u, dtype=complex)
    beta = p.kappa - p.rho * p.xi * 1j * u
    d = np.sqrt(beta**2 + p.xi**2 * (1j * u + u**2))
    g = (beta - d) / (beta + d)
    decay = np.exp(-d * t)
    c = (p.kappa * p.theta / p.xi**2) * (
        (beta - d) * t - 2.0 * np.log((1.0 - g * decay) / (1.0 - g))
    )
    dd = (beta - d) / p.xi**2 * (1.0 - decay) / (1.0 - g * decay)
    return c, dd


def char_func(u, t, p: HestonParams):
    """E[exp(i u X)] for X = ln(S_t / F_0) under the forward measure (any complex ``u``)."""
    c, d = _exponent_terms(u, t, p)
    return np.exp(c + d * p.v0)


def _lewis_integrand(u, x, t, p: HestonParams):
    value = np.exp(1j * u * x) * char_func(u - 0.5j, t, p)
    return value.real / (u * u + 0.25)


def call_price(spot, strike, t, p: HestonParams, r=0.0, q=0.0) -> float:
    """European call price by numerical integration (scalar inputs)."""
    fwd = float(spot) * np.exp((r - q) * t)
    x = np.log(fwd / strike)
    integral, _ = quad(_lewis_integrand, 0.0, np.inf, args=(x, t, p), limit=500, epsabs=1e-13)
    return float(np.exp(-r * t) * (fwd - np.sqrt(fwd * strike) / np.pi * integral))


def put_price(spot, strike, t, p: HestonParams, r=0.0, q=0.0) -> float:
    c = call_price(spot, strike, t, p, r, q)
    return c - spot * np.exp(-q * t) + strike * np.exp(-r * t)


def implied_vols(spot, strikes, t, p: HestonParams, r=0.0, q=0.0):
    """Black-Scholes implied volatilities of the model at the given strikes.

    The out-of-the-money option is priced and inverted, so far wings keep their accuracy.
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    fwd = spot * np.exp((r - q) * t)
    out = np.empty_like(strikes)
    for i, k in enumerate(strikes):
        if k >= fwd:
            out[i] = implied_vol(call_price(spot, k, t, p, r, q), spot, k, t, r, q, call=True)
        else:
            out[i] = implied_vol(put_price(spot, k, t, p, r, q), spot, k, t, r, q, call=False)
    return out


class VectorPricer:
    """Heston call prices for many (spot, v0) pairs at once, on a fixed Gauss-Legendre grid.

    ``C`` and ``D`` depend only on the expiry and on the parameters other than ``v0``, so they are
    computed once. Pricing ``n`` states then costs ``n * nodes`` complex exponentials. The
    integration range ends where the integrand has fallen below ``1e-15`` for the smallest
    variance ``v_min`` that will be priced (the Heston characteristic function decays only
    exponentially in ``u``, so a Gaussian rule of thumb would cut the range too early).
    """

    def __init__(
        self,
        p: HestonParams,
        t: float,
        strike: float,
        r=0.0,
        q=0.0,
        nodes=600,
        v_min=None,
        max_cutoff=1e4,
    ):
        self.p, self.t, self.strike, self.r, self.q = p, float(t), float(strike), r, q
        v_min = max(0.25 * min(p.v0, p.theta), 1e-5) if v_min is None else v_min
        probe = np.geomspace(1.0, max_cutoff, 4000)
        c, d = _exponent_terms(probe - 0.5j, self.t, p)
        size = np.abs(np.exp(c + d * v_min)) / (probe**2 + 0.25)
        significant = np.flatnonzero(size > 1e-15)
        upper = 1.2 * probe[significant[-1]] if significant.size else probe[0]
        z, w = np.polynomial.legendre.leggauss(nodes)
        self.u = 0.5 * upper * (z + 1.0)
        self.w = 0.5 * upper * w
        c, d = _exponent_terms(self.u - 0.5j, self.t, p)
        self._c, self._d = c, d
        self._denominator = self.u**2 + 0.25
        self.upper = upper

    def call(self, spot, v0):
        spot = np.atleast_1d(np.asarray(spot, dtype=float))
        v0 = np.broadcast_to(np.asarray(v0, dtype=float), spot.shape)
        fwd = spot * np.exp((self.r - self.q) * self.t)
        x = np.log(fwd / self.strike)
        psi = np.exp(self._c[None, :] + self._d[None, :] * v0[:, None])
        integrand = (np.exp(1j * self.u[None, :] * x[:, None]) * psi).real / self._denominator
        integral = integrand @ self.w
        return np.exp(-self.r * self.t) * (fwd - np.sqrt(fwd * self.strike) / np.pi * integral)

    def delta_and_vega(self, spot, v0, rel_bump=1e-3):
        """Model delta dC/dS and sensitivity dC/dv0 by central differences."""
        spot = np.atleast_1d(np.asarray(spot, dtype=float))
        v0 = np.broadcast_to(np.asarray(v0, dtype=float), spot.shape)
        up, dn = spot * (1 + rel_bump), spot * (1 - rel_bump)
        delta = (self.call(up, v0) - self.call(dn, v0)) / (up - dn)
        dv = 0.05 * v0 + 1e-6
        v_up, v_dn = v0 + dv, np.maximum(v0 - dv, 1e-8)
        dcdv = (self.call(spot, v_up) - self.call(spot, v_dn)) / (v_up - v_dn)
        return delta, dcdv


def simulate(
    p: HestonParams,
    spot0: float,
    t: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
    r: float = 0.0,
    q: float = 0.0,
    substeps: int = 4,
    psi_critical: float = 1.5,
):
    """Paths of (S, v) on ``n_steps`` equal steps, each cut into ``substeps`` QE steps.

    Returns arrays of shape ``(n_paths, n_steps + 1)``.
    """
    dt = t / (n_steps * substeps)
    decay = np.exp(-p.kappa * dt)
    k0 = -p.rho * p.kappa * p.theta * dt / p.xi
    drift_term = 0.5 * dt * (p.kappa * p.rho / p.xi - 0.5)
    k1 = drift_term - p.rho / p.xi
    k2 = drift_term + p.rho / p.xi
    k3 = k4 = 0.5 * dt * (1.0 - p.rho**2)

    log_s = np.full(n_paths, np.log(spot0))
    v = np.full(n_paths, p.v0)
    s_out = np.empty((n_paths, n_steps + 1))
    v_out = np.empty((n_paths, n_steps + 1))
    s_out[:, 0], v_out[:, 0] = spot0, p.v0
    for step in range(1, n_steps + 1):
        for _ in range(substeps):
            m = p.theta + (v - p.theta) * decay
            s2 = v * p.xi**2 * decay * (1.0 - decay) / p.kappa + p.theta * p.xi**2 * (
                1.0 - decay
            ) ** 2 / (2.0 * p.kappa)
            psi = s2 / m**2
            quadratic = psi <= psi_critical

            inv = 2.0 / np.maximum(psi, 1e-12)
            b2 = np.maximum(inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0)), 0.0)
            v_quad = m / (1.0 + b2) * (np.sqrt(b2) + rng.standard_normal(n_paths)) ** 2

            prob_zero = (psi - 1.0) / (psi + 1.0)
            rate = (1.0 - prob_zero) / m
            u = rng.random(n_paths)
            v_exp = np.where(u <= prob_zero, 0.0, np.log((1.0 - prob_zero) / (1.0 - u)) / rate)

            v_new = np.where(quadratic, v_quad, v_exp)
            log_s = (
                log_s
                + (r - q) * dt
                + k0
                + k1 * v
                + k2 * v_new
                + np.sqrt(np.maximum(k3 * v + k4 * v_new, 0.0)) * rng.standard_normal(n_paths)
            )
            v = v_new
        s_out[:, step] = np.exp(log_s)
        v_out[:, step] = v
    return s_out, v_out
