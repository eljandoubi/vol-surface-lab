import numpy as np
import pytest

from volsurf import bs

GRID = [
    # spot, strike, t, vol, r, q
    (100.0, 100.0, 1.0, 0.20, 0.05, 0.00),
    (100.0, 80.0, 0.5, 0.35, 0.01, 0.02),
    (100.0, 130.0, 0.25, 0.60, 0.00, 0.00),
    (50.0, 55.0, 2.0, 0.15, 0.03, 0.01),
    (3.0, 2.5, 0.05, 0.9, 0.0, 0.0),
]


def test_reference_values():
    # Hull-style textbook case: S=K=100, T=1, vol 20%, r 5%, no yield.
    call = bs.price(100, 100, 1.0, 0.2, r=0.05, call=True)
    put = bs.price(100, 100, 1.0, 0.2, r=0.05, call=False)
    assert call == pytest.approx(10.450583572185565, abs=1e-10)
    assert put == pytest.approx(5.573526022256971, abs=1e-10)


@pytest.mark.parametrize("spot,strike,t,vol,r,q", GRID)
def test_put_call_parity(spot, strike, t, vol, r, q):
    c = bs.price(spot, strike, t, vol, r, q, call=True)
    p = bs.price(spot, strike, t, vol, r, q, call=False)
    expected = spot * np.exp(-q * t) - strike * np.exp(-r * t)
    assert c - p == pytest.approx(expected, abs=1e-12 * max(spot, strike))


def test_price_is_increasing_in_vol_and_bounded():
    vols = np.linspace(0.05, 2.0, 40)
    c = bs.price(100, 110, 0.75, vols, r=0.02)
    assert np.all(np.diff(c) > 0)
    assert np.all(c < 100)
    assert np.all(c > np.maximum(100 - 110 * np.exp(-0.02 * 0.75), 0))


def test_zero_vol_gives_forward_intrinsic_value():
    fwd = 100 * np.exp(0.03 * 0.5)
    df = np.exp(-0.03 * 0.5)
    assert bs.price(100, 90, 0.5, 0.0, r=0.03) == pytest.approx(df * (fwd - 90))
    assert bs.price(100, 90, 0.5, 0.0, r=0.03, call=False) == 0.0
    assert bs.price(100, 100, 0.5, 0.0, r=0.0) == 0.0


def test_shapes_broadcast():
    strikes = np.linspace(80, 120, 9)[:, None]
    ts = np.array([0.1, 0.5, 1.0])[None, :]
    out = bs.price(100, strikes, ts, 0.3)
    assert out.shape == (9, 3)
    g = bs.greeks(100, strikes, ts, 0.3)
    assert g.vega.shape == (9, 3)


def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


@pytest.mark.parametrize("call", [True, False])
@pytest.mark.parametrize("spot,strike,t,vol,r,q", GRID)
def test_first_order_greeks_match_finite_differences(spot, strike, t, vol, r, q, call):
    g = bs.greeks(spot, strike, t, vol, r, q, call=call)
    h_s, h_v, h_t, h_r = 1e-4 * spot, 1e-5, 1e-6, 1e-6
    delta = _fd(lambda s: bs.price(s, strike, t, vol, r, q, call), spot, h_s)
    vega = _fd(lambda v: bs.price(spot, strike, t, v, r, q, call), vol, h_v)
    theta = -_fd(lambda tt: bs.price(spot, strike, tt, vol, r, q, call), t, h_t)
    rho = _fd(lambda rr: bs.price(spot, strike, t, vol, rr, q, call), r, h_r)
    assert g.delta == pytest.approx(delta, rel=1e-6, abs=1e-8)
    assert g.vega == pytest.approx(vega, rel=1e-6, abs=1e-8)
    assert g.theta == pytest.approx(theta, rel=1e-5, abs=1e-6)
    assert g.rho == pytest.approx(rho, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("spot,strike,t,vol,r,q", GRID)
def test_second_order_greeks_match_finite_differences(spot, strike, t, vol, r, q):
    g = bs.greeks(spot, strike, t, vol, r, q)
    h_s, h_v = 1e-3 * spot, 1e-4

    def value(s, v):
        return bs.price(s, strike, t, v, r, q)

    gamma = (value(spot + h_s, vol) - 2 * value(spot, vol) + value(spot - h_s, vol)) / h_s**2
    volga = (value(spot, vol + h_v) - 2 * value(spot, vol) + value(spot, vol - h_v)) / h_v**2
    vanna = (
        value(spot + h_s, vol + h_v)
        - value(spot + h_s, vol - h_v)
        - value(spot - h_s, vol + h_v)
        + value(spot - h_s, vol - h_v)
    ) / (4 * h_s * h_v)
    assert g.gamma == pytest.approx(gamma, rel=1e-4)
    assert g.volga == pytest.approx(volga, rel=1e-3, abs=1e-5)
    assert g.vanna == pytest.approx(vanna, rel=1e-3, abs=1e-5)


def test_put_delta_is_call_delta_minus_discounted_one():
    gc = bs.greeks(100, 95, 0.4, 0.3, 0.02, 0.01, call=True)
    gp = bs.greeks(100, 95, 0.4, 0.3, 0.02, 0.01, call=False)
    assert gc.delta - gp.delta == pytest.approx(np.exp(-0.01 * 0.4))
    assert gc.gamma == pytest.approx(gp.gamma)
    assert gc.vega == pytest.approx(gp.vega)


def test_forward_helper():
    assert bs.forward(100, 2.0, r=0.03, q=0.01) == pytest.approx(100 * np.exp(0.04))
