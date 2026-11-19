import numpy as np
import pytest

from volsurf import bs
from volsurf.svi import SviSlice, butterfly_violations, fit_slice

BENIGN = SviSlice(a=0.02, b=0.25, rho=-0.4, m=0.03, sigma=0.2)


def test_total_variance_formula_and_minimum():
    k = np.linspace(-1, 1, 41)
    s = BENIGN
    expected = s.a + s.b * (s.rho * (k - s.m) + np.sqrt((k - s.m) ** 2 + s.sigma**2))
    np.testing.assert_allclose(s.total_variance(k), expected)
    grid = np.linspace(-3, 3, 600001)
    assert s.total_variance(grid).min() == pytest.approx(s.minimum_total_variance, abs=1e-9)


def test_derivatives_match_finite_differences():
    k = np.linspace(-0.8, 0.8, 17)
    h = 1e-5
    s = BENIGN
    slope = (s.total_variance(k + h) - s.total_variance(k - h)) / (2 * h)
    curvature = (s.total_variance(k + h) - 2 * s.total_variance(k) + s.total_variance(k - h)) / h**2
    np.testing.assert_allclose(s.slope(k), slope, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(s.curvature(k), curvature, rtol=1e-4, atol=1e-6)


def test_admissibility_and_lee_bound():
    assert BENIGN.is_admissible()
    assert BENIGN.satisfies_lee_bound()
    assert not SviSlice(0.02, -0.1, 0.0, 0.0, 0.1).is_admissible()
    assert not SviSlice(0.02, 0.1, 1.0, 0.0, 0.1).is_admissible()
    assert not SviSlice(-0.5, 0.1, 0.0, 0.0, 0.1).is_admissible()  # negative minimum variance
    assert not SviSlice(0.0, 1.5, 0.5, 0.0, 0.1).satisfies_lee_bound()  # right slope 2.25 > 2
    assert SviSlice(0.0, 1.0, 0.99, 0.0, 0.1).satisfies_lee_bound()  # right slope 1.99
    assert not SviSlice(0.0, 1.01, 0.99, 0.0, 0.1).satisfies_lee_bound()  # right slope 2.0099


def test_density_formula_matches_second_derivative_of_black_scholes_prices():
    # F = 1 and t = 1, so total variance equals the squared Black-Scholes volatility.
    for slice_ in (BENIGN, SviSlice(0.01, 0.4, 0.3, -0.05, 0.15)):
        k = np.linspace(-0.6, 0.6, 13)
        strike = np.exp(k)
        h = 1e-4 * strike

        def call(kk, slice_=slice_):
            return bs.price(1.0, kk, 1.0, np.sqrt(slice_.total_variance(np.log(kk))))

        second = (call(strike + h) - 2 * call(strike) + call(strike - h)) / h**2
        numeric = strike * second
        np.testing.assert_allclose(slice_.density(k), numeric, rtol=2e-5, atol=1e-8)


def test_density_integrates_to_one_and_has_forward_mean_one():
    k = np.linspace(-8, 8, 200001)
    p = BENIGN.density(k)
    dk = k[1] - k[0]
    assert np.all(p > 0)
    assert p.sum() * dk == pytest.approx(1.0, abs=1e-4)
    assert (np.exp(k) * p).sum() * dk == pytest.approx(1.0, abs=1e-4)


def test_benign_slice_has_no_butterfly_arbitrage():
    k = np.linspace(-4, 4, 4001)
    assert np.all(BENIGN.butterfly_g(k) > 0)
    assert butterfly_violations(BENIGN, k).size == 0


def test_steep_slice_violates_the_butterfly_condition_and_has_negative_density():
    bad = SviSlice(a=0.0005, b=0.9, rho=0.9, m=0.0, sigma=0.02)
    assert bad.is_admissible()
    k = np.linspace(-0.5, 0.5, 2001)
    violations = butterfly_violations(bad, k)
    assert violations.size > 0
    assert np.all(bad.density(violations) < 0)
    # the same negative density shows up in the second derivative of the call price in strike
    k0 = violations[len(violations) // 2]
    strike, h = np.exp(k0), 1e-5 * np.exp(k0)

    def call(kk):
        return bs.price(1.0, kk, 1.0, np.sqrt(bad.total_variance(np.log(kk))))

    assert (call(strike + h) - 2 * call(strike) + call(strike - h)) / h**2 < 0


@pytest.mark.parametrize(
    "truth",
    [
        BENIGN,
        SviSlice(0.005, 0.12, 0.1, 0.0, 0.08),
        SviSlice(0.03, 0.5, -0.7, -0.1, 0.3),
        SviSlice(0.0, 0.08, -0.2, 0.05, 0.05),
    ],
)
def test_fit_reproduces_noiseless_data(truth):
    k = np.linspace(-0.9, 0.9, 31)
    w = truth.total_variance(k)
    fit = fit_slice(k, w)
    assert fit.max_abs_error < 1e-7
    assert fit.slice.is_admissible()
    np.testing.assert_allclose(fit.slice.total_variance(k), w, atol=1e-7)


def test_fit_is_robust_to_noise_and_to_sparse_strikes():
    rng = np.random.default_rng(5)
    k = np.sort(rng.uniform(-0.7, 0.7, 11))
    clean = BENIGN.total_variance(k)
    noisy = clean + rng.normal(0, 2e-4, k.size)
    fit = fit_slice(k, noisy)
    assert fit.rmse < 4e-4
    grid = np.linspace(-0.6, 0.6, 50)
    assert np.max(np.abs(fit.slice.total_variance(grid) - BENIGN.total_variance(grid))) < 2e-3


def test_weights_pull_the_fit_towards_heavily_weighted_points():
    k = np.linspace(-0.5, 0.5, 21)
    w = BENIGN.total_variance(k).copy()
    w[10] += 0.01  # one outlier at the money
    flat = fit_slice(k, w)
    weights = np.ones_like(k)
    weights[10] = 1e-6
    robust = fit_slice(k, w, weights=weights)
    assert abs(robust.slice.total_variance(k[10]) - BENIGN.total_variance(k[10])) < abs(
        flat.slice.total_variance(k[10]) - BENIGN.total_variance(k[10])
    )


def test_fit_validates_input():
    with pytest.raises(ValueError):
        fit_slice([0.0, 0.1], [0.04, 0.05])
    with pytest.raises(ValueError):
        fit_slice(np.zeros((3, 3)), np.zeros((3, 3)))
