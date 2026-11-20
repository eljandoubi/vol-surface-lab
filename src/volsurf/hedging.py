"""Delta hedging of a short call in a Heston world, with four ways to choose the hedge ratio.

The hedger sells one at-the-money call at its model price and rebalances the underlying on a fixed
schedule, financing at a zero rate. At expiry the profit and loss of one path is

    pnl = c0 - max(S_T - K, 0) + sum_i delta_i * (S_{i+1} - S_i).

The hedge ratios compared are

* ``bs_entry_vol``: the Black-Scholes delta at the implied volatility seen on day 0,
* ``bs_current_vol``: the Black-Scholes delta at the implied volatility the model gives today,
* ``heston_delta``: the model delta dC/dS at the current spot and variance,
* ``heston_min_variance``: the model delta plus ``rho * xi / S * dC/dv``, the part of the variance
  move that is correlated with the spot move.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bs
from .heston import HestonParams, VectorPricer, call_price, implied_vols, simulate
from .iv import implied_vol

STRATEGIES = ("none", "bs_entry_vol", "bs_current_vol", "heston_delta", "heston_min_variance")


@dataclass(frozen=True)
class HedgeResult:
    pnl: dict[str, np.ndarray]
    entry_price: float
    entry_vol: float
    missing_implied_vol_fraction: float


def hedge_ratios(p: HestonParams, spot, variance, strike, tau, entry_vol):
    """The four hedge ratios for every path, plus the fraction of states without implied vol."""
    pricer = VectorPricer(p, tau, strike, v_min=max(float(np.min(variance)), 1e-3))
    delta, dcdv = pricer.delta_and_vega(spot, variance)
    price = pricer.call(spot, variance)
    current = implied_vol(price, spot, strike, tau)
    missing = np.isnan(current)
    current = np.where(missing, entry_vol, current)
    return {
        "bs_entry_vol": bs.greeks(spot, strike, tau, entry_vol).delta,
        "bs_current_vol": bs.greeks(spot, strike, tau, current).delta,
        "heston_delta": delta,
        "heston_min_variance": delta + p.rho * p.xi / spot * dcdv,
    }, float(missing.mean())


def run_hedge(
    p: HestonParams,
    days: int = 30,
    rebalances_per_day: int = 1,
    n_paths: int = 5000,
    seed: int = 0,
    spot0: float = 100.0,
    strike: float = 100.0,
    year_days: float = 365.0,
) -> HedgeResult:
    t_total = days / year_days
    n_steps = days * rebalances_per_day
    rng = np.random.default_rng(seed)
    spot, variance = simulate(p, spot0, t_total, n_steps, n_paths, rng)
    dt = t_total / n_steps

    entry_price = call_price(spot0, strike, t_total, p)
    entry_vol = float(implied_vols(spot0, [strike], t_total, p)[0])

    gains = {name: np.zeros(n_paths) for name in STRATEGIES[1:]}
    missing = []
    for i in range(n_steps):
        tau = t_total - i * dt
        ratios, miss = hedge_ratios(p, spot[:, i], variance[:, i], strike, tau, entry_vol)
        missing.append(miss)
        move = spot[:, i + 1] - spot[:, i]
        for name, ratio in ratios.items():
            gains[name] += ratio * move

    payoff = np.maximum(spot[:, -1] - strike, 0.0)
    base = entry_price - payoff
    pnl = {"none": base}
    pnl.update({name: base + gain for name, gain in gains.items()})
    return HedgeResult(pnl, float(entry_price), entry_vol, float(np.mean(missing)))


def summarise(result: HedgeResult) -> dict[str, dict[str, float]]:
    """Mean and spread of the profit and loss per strategy, in units of the option premium."""
    out = {}
    n = len(next(iter(result.pnl.values())))
    for name, pnl in result.pnl.items():
        std = float(pnl.std(ddof=1))
        out[name] = {
            "mean_over_premium": float(pnl.mean() / result.entry_price),
            "mean_standard_error": float(std / np.sqrt(n) / result.entry_price),
            "std_over_premium": float(std / result.entry_price),
            "std_standard_error": float(std / np.sqrt(2 * (n - 1)) / result.entry_price),
            "p05_over_premium": float(np.quantile(pnl, 0.05) / result.entry_price),
            "p95_over_premium": float(np.quantile(pnl, 0.95) / result.entry_price),
        }
    return out


def paired_std_ratio(a, b, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """Ratio of the standard deviations of two profit-and-loss samples on the same paths.

    The two samples come from the same simulated paths and are strongly correlated, so the paths are
    resampled together. Returns the ratio ``std(a) / std(b)`` and the percentile interval.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("need two samples of the same length")
    rng = np.random.default_rng(seed)
    n = len(a)
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ratios[i] = a[idx].std(ddof=1) / b[idx].std(ddof=1)
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(ratios, [tail, 100 - tail])
    return float(a.std(ddof=1) / b.std(ddof=1)), float(lo), float(hi)
