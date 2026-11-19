import numpy as np
import pytest

from volsurf import bs
from volsurf.iv import implied_vol


@pytest.mark.parametrize("call", [True, False])
def test_round_trip_over_a_wide_grid(call):
    rng = np.random.default_rng(0)
    n = 4000
    spot = rng.uniform(50, 150, n)
    log_m = rng.uniform(-1.0, 1.0, n)
    strike = spot * np.exp(log_m)
    t = rng.uniform(0.02, 3.0, n)
    vol = rng.uniform(0.05, 2.0, n)
    r = rng.uniform(-0.01, 0.06, n)
    q = rng.uniform(0.0, 0.04, n)
    p = bs.price(spot, strike, t, vol, r, q, call=call)
    est = implied_vol(p, spot, strike, t, r, q, call=call)

    df = np.exp(-r * t)
    fwd = spot * np.exp((r - q) * t)
    intrinsic = df * (np.maximum(fwd - strike, 0) if call else np.maximum(strike - fwd, 0))
    time_value = (p - intrinsic) / (df * np.sqrt(fwd * strike))

    # NaN only where rounding left no time value to invert.
    ok = np.isfinite(est)
    assert np.all(time_value[~ok] < 1e-10)

    # Every returned volatility reprices the option to rounding error.
    repriced = bs.price(spot[ok], strike[ok], t[ok], est[ok], r[ok], q[ok], call=call)
    assert np.max(np.abs(repriced - p[ok]) / spot[ok]) < 1e-11

    # Where the time value is not lost to rounding the volatility itself is recovered.
    well_conditioned = ok & (time_value > 1e-6)
    assert well_conditioned.mean() > 0.9
    np.testing.assert_allclose(est[well_conditioned], vol[well_conditioned], rtol=1e-7, atol=1e-9)


def test_mixed_calls_and_puts_in_one_call():
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    is_call = strikes >= 100
    p = bs.price(100.0, strikes, 0.5, 0.45, call=is_call)
    est = implied_vol(p, 100.0, strikes, 0.5, call=is_call)
    np.testing.assert_allclose(est, 0.45, atol=1e-10)


@pytest.mark.parametrize("log_moneyness", [1.2, 3.0])
def test_deep_out_of_the_money_tail_keeps_relative_accuracy(log_moneyness):
    strike = 100.0 * np.exp(log_moneyness)
    p = bs.price(100.0, strike, 0.1, 0.4)
    assert 0 < p < 1e-6
    assert implied_vol(p, 100.0, strike, 0.1) == pytest.approx(0.4, rel=1e-8)
    # the same distance on the put side, where the strike is below the forward
    put_strike = 100.0 * np.exp(-log_moneyness)
    q = bs.price(100.0, put_strike, 0.1, 0.4, call=False)
    assert 0 < q < 1e-6
    assert implied_vol(q, 100.0, put_strike, 0.1, call=False) == pytest.approx(0.4, rel=1e-8)


def test_in_the_money_option_with_small_time_value():
    p = bs.price(100.0, 85.0, 0.25, 0.2)
    time_value = p - 15.0
    assert 1e-4 < time_value < 1.0
    assert implied_vol(p, 100.0, 85.0, 0.25) == pytest.approx(0.2, rel=1e-8)


def test_in_the_money_price_that_rounds_to_intrinsic_has_no_volatility():
    p = bs.price(100.0, 70.0, 0.05, 0.2)
    assert abs(p - 30.0) < 1e-12
    assert np.isnan(implied_vol(p, 100.0, 70.0, 0.05))


def test_prices_outside_the_no_arbitrage_band_return_nan():
    fwd = 100.0
    intrinsic = 10.0
    prices = np.array([intrinsic - 1e-6, intrinsic, fwd, fwd + 1.0, 0.0, -1.0, np.nan])
    out = implied_vol(prices, fwd, 90.0, 1.0)
    assert np.all(np.isnan(out))


def test_non_positive_expiry_returns_nan():
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, 0.0))
    assert np.isnan(implied_vol(5.0, 100.0, 100.0, -1.0))


def test_valid_and_invalid_entries_do_not_contaminate_each_other():
    p = np.array([bs.price(100.0, 100.0, 1.0, 0.3), -1.0, bs.price(100.0, 120.0, 1.0, 0.6)])
    out = implied_vol(p, 100.0, np.array([100.0, 100.0, 120.0]), 1.0)
    assert out[0] == pytest.approx(0.3, abs=1e-10)
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(0.6, abs=1e-10)


def test_scalar_in_scalar_out():
    out = implied_vol(bs.price(100, 105, 0.3, 0.25, r=0.01), 100, 105, 0.3, r=0.01)
    assert np.ndim(out) == 0
    assert float(out) == pytest.approx(0.25, abs=1e-10)


def test_high_volatility_and_long_expiry():
    p = bs.price(100.0, 100.0, 10.0, 3.0)
    assert implied_vol(p, 100.0, 100.0, 10.0) == pytest.approx(3.0, rel=1e-6)
