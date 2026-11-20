"""Coin-margined (inverse) European options, priced in units of the underlying coin.

An inverse call on a dollar-quoted coin pays ``max(S_T - K, 0) / S_T`` coins at expiry, which is
worth ``max(S_T - K, 0)`` dollars at that time: the vanilla payoff. Its premium in coins is the
dollar price divided by today's spot, and the Greeks in coins follow by the quotient rule. Spot
moves are in dollars per coin, so ``delta`` is coins of premium per dollar of spot.

Taking the coin as numeraire turns the dollar into the underlying: with zero rates the inverse call
equals ``K`` times a put on ``1 / S`` struck at ``1 / K`` with the same volatility (the symmetry of
Garman and Kohlhagen between the two currencies). The tests check this independently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bs


def inverse_price(spot, strike, t, vol, r=0.0, q=0.0, call=True):
    """Premium in coins of a coin-margined option."""
    return bs.price(spot, strike, t, vol, r, q, call) / np.asarray(spot, dtype=float)


@dataclass(frozen=True)
class InverseGreeks:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray


def inverse_greeks(spot, strike, t, vol, r=0.0, q=0.0, call=True) -> InverseGreeks:
    """Premium, delta, gamma and vega in coins (vega per unit of volatility)."""
    spot = np.asarray(spot, dtype=float)
    dollar = bs.greeks(spot, strike, t, vol, r, q, call)
    coins = bs.price(spot, strike, t, vol, r, q, call) / spot
    delta = (dollar.delta - coins) / spot
    gamma = dollar.gamma / spot - 2.0 * (dollar.delta - coins) / spot**2
    return InverseGreeks(coins, delta, gamma, dollar.vega / spot)
