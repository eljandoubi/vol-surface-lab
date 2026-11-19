import numpy as np
import pytest

from volsurf import bs, heston
from volsurf.heston import HestonParams, VectorPricer

PARAMS = [
    HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7),
    HestonParams(v0=0.09, kappa=1.0, theta=0.16, xi=0.9, rho=-0.3),  # Feller violated
    HestonParams(v0=0.25, kappa=4.0, theta=0.20, xi=1.2, rho=0.2),
]


def test_parameter_validation():
    with pytest.raises(ValueError):
        HestonParams(0.04, 2.0, 0.04, 0.5, 1.0)
    with pytest.raises(ValueError):
        HestonParams(-0.01, 2.0, 0.04, 0.5, -0.5)
    with pytest.raises(ValueError):
        HestonParams(0.04, 2.0, 0.04, 0.0, -0.5)
    assert HestonParams(0.04, 2.0, 0.04, 0.3, -0.5).feller > 0
    assert HestonParams(0.04, 2.0, 0.04, 0.5, -0.5).feller < 0
    assert PARAMS[1].feller < 0  # the second parameter set violates the Feller condition


@pytest.mark.parametrize("p", PARAMS)
@pytest.mark.parametrize("t", [0.05, 0.5, 3.0])
def test_char_function_is_a_valid_martingale_transform(p, t):
    assert heston.char_func(0.0, t, p) == pytest.approx(1.0, abs=1e-14)
    # E[S_t / F_0] = 1 is the statement psi(-i) = 1
    assert heston.char_func(-1j, t, p) == pytest.approx(1.0, abs=1e-10)
    u = np.linspace(0.1, 40, 200)
    assert np.all(np.abs(heston.char_func(u, t, p)) <= 1.0 + 1e-12)


def test_char_function_has_hermitian_symmetry():
    p = PARAMS[0]
    u = np.array([0.3, 2.0, 11.0])
    np.testing.assert_allclose(heston.char_func(-u, 0.7, p), np.conj(heston.char_func(u, 0.7, p)))


def _gap_to_black_scholes(xi, rho, strike=70.0):
    p = HestonParams(v0=0.09, kappa=1.5, theta=0.09, xi=xi, rho=rho)
    h = heston.call_price(100.0, strike, 0.8, p, r=0.02, q=0.01)
    return h - bs.price(100.0, strike, 0.8, 0.3, r=0.02, q=0.01)


def test_vanishing_vol_of_vol_recovers_black_scholes():
    for strike in (70.0, 100.0, 140.0):
        assert abs(_gap_to_black_scholes(1e-3, 0.0, strike)) < 5e-6


def test_gap_to_black_scholes_is_second_order_without_correlation_and_first_order_with_it():
    # Without correlation the smile is symmetric and the price moves with xi squared.
    ratio = _gap_to_black_scholes(1e-1, 0.0) / _gap_to_black_scholes(1e-2, 0.0)
    assert ratio == pytest.approx(100.0, rel=0.05)
    # With correlation the skew appears and the price moves linearly with xi.
    ratio = _gap_to_black_scholes(1e-2, -0.5) / _gap_to_black_scholes(1e-3, -0.5)
    assert ratio == pytest.approx(10.0, rel=0.05)


@pytest.mark.parametrize("p", PARAMS)
def test_put_call_parity_and_bounds(p):
    for strike in (60.0, 100.0, 170.0):
        c = heston.call_price(100.0, strike, 0.6, p, r=0.01, q=0.02)
        pu = heston.put_price(100.0, strike, 0.6, p, r=0.01, q=0.02)
        assert c - pu == pytest.approx(100 * np.exp(-0.02 * 0.6) - strike * np.exp(-0.01 * 0.6))
        assert max(100 * np.exp(-0.02 * 0.6) - strike * np.exp(-0.01 * 0.6), 0) <= c
        assert c <= 100 * np.exp(-0.02 * 0.6)


@pytest.mark.parametrize("p", PARAMS)
def test_grid_pricer_agrees_with_adaptive_quadrature(p):
    strikes = [70.0, 100.0, 125.0]
    spots = np.array([90.0, 100.0, 112.0])
    for strike in strikes:
        pricer = VectorPricer(p, 0.4, strike, r=0.01, q=0.0)
        fast = pricer.call(spots, p.v0)
        slow = [heston.call_price(s, strike, 0.4, p, r=0.01) for s in spots]
        np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-8)


def test_grid_pricer_accepts_different_variances_per_state():
    p = PARAMS[0]
    pricer = VectorPricer(p, 0.25, 100.0)
    v = np.array([0.01, 0.04, 0.2])
    fast = pricer.call(np.array([100.0, 100.0, 100.0]), v)
    slow = [heston.call_price(100.0, 100.0, 0.25, p.with_v0(x)) for x in v]
    np.testing.assert_allclose(fast, slow, rtol=1e-6)


def test_model_delta_and_variance_sensitivity_signs():
    p = PARAMS[0]
    pricer = VectorPricer(p, 0.25, 100.0)
    delta, dcdv = pricer.delta_and_vega(np.array([90.0, 100.0, 110.0]), 0.04)
    assert np.all((0 < delta) & (delta < 1))
    assert np.all(np.diff(delta) > 0)
    assert np.all(dcdv > 0)


@pytest.mark.parametrize("p", PARAMS[:2])
def test_monte_carlo_agrees_with_the_semi_analytic_price(p):
    rng = np.random.default_rng(7)
    s, _ = heston.simulate(p, 100.0, 0.5, n_steps=10, n_paths=60_000, rng=rng, r=0.01, q=0.005)
    final = s[:, -1]
    for strike in (85.0, 100.0, 120.0):
        payoff = np.exp(-0.01 * 0.5) * np.maximum(final - strike, 0.0)
        mc = payoff.mean()
        se = payoff.std(ddof=1) / np.sqrt(len(payoff))
        exact = heston.call_price(100.0, strike, 0.5, p, r=0.01, q=0.005)
        assert abs(mc - exact) < 4 * se + 0.02


def test_simulator_moments():
    p = PARAMS[0]
    rng = np.random.default_rng(3)
    t = 1.0
    s, v = heston.simulate(p, 100.0, t, n_steps=12, n_paths=80_000, rng=rng, r=0.03, q=0.01)
    forward = 100.0 * np.exp(0.02 * t)
    se = s[:, -1].std(ddof=1) / np.sqrt(s.shape[0])
    assert abs(s[:, -1].mean() - forward) < 4 * se + 0.05
    expected_v = p.theta + (p.v0 - p.theta) * np.exp(-p.kappa * t)
    assert v[:, -1].mean() == pytest.approx(expected_v, abs=4 * v[:, -1].std() / np.sqrt(len(v)))
    assert np.all(v >= 0)
    assert np.all(s > 0)


def test_simulator_shapes_and_reproducibility():
    p = PARAMS[2]
    a = heston.simulate(p, 50.0, 0.2, 5, 100, np.random.default_rng(1))
    b = heston.simulate(p, 50.0, 0.2, 5, 100, np.random.default_rng(1))
    assert a[0].shape == (100, 6)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert np.all(a[0][:, 0] == 50.0)
    assert np.all(a[1][:, 0] == p.v0)


def test_smile_shape_follows_the_correlation_and_vol_of_vol():
    base = HestonParams(0.04, 2.0, 0.04, 0.6, -0.7)
    strikes = np.array([80.0, 100.0, 120.0])
    iv = heston.implied_vols(100.0, strikes, 0.5, base)
    assert iv[0] > iv[1] > iv[2]  # negative correlation: downside skew
    flat = heston.implied_vols(100.0, strikes, 0.5, HestonParams(0.04, 2.0, 0.04, 0.05, 0.0))
    assert np.ptp(flat) < 0.01
    assert flat[1] == pytest.approx(0.2, abs=1e-3)  # no vol-of-vol: ATM volatility is sqrt(v0)
    smile = heston.implied_vols(100.0, strikes, 0.5, HestonParams(0.04, 2.0, 0.04, 0.9, 0.0))
    assert smile[0] > smile[1] and smile[2] > smile[1]  # zero correlation: symmetric smile
