import numpy as np
import pytest

from volsurf import bs
from volsurf.inverse import inverse_greeks, inverse_price

CASES = [
    (100.0, 100.0, 0.5, 0.8, True),
    (100.0, 130.0, 0.25, 0.6, True),
    (100.0, 70.0, 1.0, 0.9, False),
    (30000.0, 28000.0, 0.1, 0.55, True),
]


@pytest.mark.parametrize("spot,strike,t,vol,call", CASES)
def test_premium_is_the_dollar_price_in_coins(spot, strike, t, vol, call):
    expected = bs.price(spot, strike, t, vol, call=call) / spot
    assert inverse_price(spot, strike, t, vol, call=call) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("spot,strike,t,vol", [c[:4] for c in CASES])
def test_inverse_call_is_k_times_a_put_on_the_reciprocal(spot, strike, t, vol):
    # currency symmetry, zero rates: c(S, K) = K * put(1/S, 1/K) with the same volatility
    dual = strike * bs.price(1.0 / spot, 1.0 / strike, t, vol, call=False)
    assert inverse_price(spot, strike, t, vol, call=True) == pytest.approx(dual, rel=1e-12)


def test_put_call_parity_in_coins():
    spot, strike, t, vol = 100.0, 120.0, 0.7, 0.7
    gap = inverse_price(spot, strike, t, vol, call=True) - inverse_price(
        spot, strike, t, vol, call=False
    )
    assert gap == pytest.approx(1.0 - strike / spot, abs=1e-14)


def test_zero_volatility_gives_the_intrinsic_value_in_coins():
    assert inverse_price(100.0, 80.0, 0.5, 0.0, call=True) == pytest.approx(1.0 - 80.0 / 100.0)
    assert inverse_price(100.0, 120.0, 0.5, 0.0, call=True) == 0.0


@pytest.mark.parametrize("spot,strike,t,vol,call", CASES)
def test_greeks_match_finite_differences(spot, strike, t, vol, call):
    g = inverse_greeks(spot, strike, t, vol, call=call)

    def premium(s=spot, v=vol):
        return inverse_price(s, strike, t, v, call=call)

    h = spot * 1e-4
    delta = (premium(spot + h) - premium(spot - h)) / (2 * h)
    gamma = (premium(spot + h) - 2 * premium() + premium(spot - h)) / h**2
    vega = (premium(v=vol + 1e-5) - premium(v=vol - 1e-5)) / 2e-5
    assert g.price == pytest.approx(premium(), rel=1e-14)
    assert g.delta == pytest.approx(delta, rel=1e-6, abs=1e-12)
    assert g.gamma == pytest.approx(gamma, rel=1e-4, abs=1e-12 / spot)
    assert g.vega == pytest.approx(vega, rel=1e-6)


def test_vectorised_inputs_broadcast():
    strikes = np.array([80.0, 100.0, 120.0])
    g = inverse_greeks(100.0, strikes, 0.5, 0.7)
    assert g.price.shape == g.delta.shape == g.gamma.shape == g.vega.shape == (3,)
    assert np.all(np.diff(g.price) < 0)
