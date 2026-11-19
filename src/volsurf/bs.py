"""Black-Scholes-Merton prices and Greeks for European options with a continuous dividend yield.

Conventions: ``t`` is the time to expiry in years, ``vol`` the annualised volatility, ``r`` the
continuously compounded rate and ``q`` the continuous yield. Vega is per unit of volatility
(multiply by 0.01 for one volatility point), theta is per year, rho per unit of rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x):
    """Standard normal density."""
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def forward(spot, t, r=0.0, q=0.0):
    """Forward price of the underlying for expiry ``t``."""
    return np.asarray(spot, dtype=float) * np.exp((np.asarray(r) - np.asarray(q)) * t)


def d1_d2(spot, strike, t, vol, r=0.0, q=0.0):
    """The two Black-Scholes arguments; NaN where ``vol * sqrt(t)`` is not positive."""
    spot, strike, t, vol, r, q = (np.asarray(x, dtype=float) for x in (spot, strike, t, vol, r, q))
    sd = vol * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (r - q) * t + 0.5 * sd**2) / sd
    return d1, d1 - sd


def price(spot, strike, t, vol, r=0.0, q=0.0, call=True):
    """Option price; a zero ``vol * sqrt(t)`` gives the discounted forward intrinsic value."""
    spot, strike, t, vol, r, q = (np.asarray(x, dtype=float) for x in (spot, strike, t, vol, r, q))
    d1, d2 = d1_d2(spot, strike, t, vol, r, q)
    df = np.exp(-r * t)
    fwd = spot * np.exp((r - q) * t)
    call_value = df * (fwd * ndtr(d1) - strike * ndtr(d2))
    put_value = df * (strike * ndtr(-d2) - fwd * ndtr(-d1))
    degenerate = ~np.isfinite(d1)
    call_value = np.where(degenerate, df * np.maximum(fwd - strike, 0.0), call_value)
    put_value = np.where(degenerate, df * np.maximum(strike - fwd, 0.0), put_value)
    return np.where(call, call_value, put_value)


@dataclass(frozen=True)
class Greeks:
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray
    vanna: np.ndarray
    volga: np.ndarray


def greeks(spot, strike, t, vol, r=0.0, q=0.0, call=True):
    """Closed-form Greeks. Vanna is d2V/(dS dvol) and volga is d2V/dvol2 (also called vomma)."""
    spot, strike, t, vol, r, q = (np.asarray(x, dtype=float) for x in (spot, strike, t, vol, r, q))
    d1, d2 = d1_d2(spot, strike, t, vol, r, q)
    sqrt_t = np.sqrt(t)
    disc_q = np.exp(-q * t)
    disc_r = np.exp(-r * t)
    pdf = norm_pdf(d1)
    sign = np.where(call, 1.0, -1.0)

    delta = np.where(call, disc_q * ndtr(d1), disc_q * (ndtr(d1) - 1.0))
    gamma = disc_q * pdf / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf * sqrt_t
    common = -spot * disc_q * pdf * vol / (2.0 * sqrt_t)
    theta = (
        common
        - sign * r * strike * disc_r * ndtr(sign * d2)
        + sign * q * spot * disc_q * ndtr(sign * d1)
    )
    rho = sign * strike * t * disc_r * ndtr(sign * d2)
    vanna = -disc_q * pdf * d2 / vol
    volga = vega * d1 * d2 / vol
    return Greeks(delta, gamma, vega, theta, rho, vanna, volga)
