import numpy as np
import pytest

from volsurf.surface import SliceQuotes, Surface, fit_surface
from volsurf.svi import SviSlice

K = np.linspace(-0.6, 0.6, 21)


def _slice_for(t, level=0.04, skew=-0.4):
    return SviSlice(
        a=0.5 * level * t, b=0.18 * np.sqrt(t), rho=skew, m=0.0, sigma=0.15 * np.sqrt(t) + 0.05
    )


def _quotes(slices):
    return [SliceQuotes(t, K, s.total_variance(K)) for t, s in slices]


def test_surface_requires_ordered_positive_expiries():
    s = _slice_for(0.5)
    with pytest.raises(ValueError):
        Surface([(0.5, s), (0.5, s)])
    with pytest.raises(ValueError):
        Surface([(0.0, s)])


def test_interpolation_is_linear_in_time_at_fixed_moneyness():
    s1, s2 = _slice_for(0.25), _slice_for(1.0)
    surface = Surface([(0.25, s1), (1.0, s2)])
    k = np.array([-0.3, 0.0, 0.4])
    t = 0.4
    weight = (t - 0.25) / 0.75
    expected = (1 - weight) * s1.total_variance(k) + weight * s2.total_variance(k)
    np.testing.assert_allclose(surface.total_variance(k, t), expected)
    np.testing.assert_allclose(surface.total_variance(k, 0.25), s1.total_variance(k))
    np.testing.assert_allclose(surface.total_variance(k, 1.0), s2.total_variance(k))


def test_extrapolation_keeps_the_end_volatilities_flat_in_time():
    s1, s2 = _slice_for(0.25), _slice_for(1.0)
    surface = Surface([(0.25, s1), (1.0, s2)])
    k = np.array([0.1])
    assert surface.implied_vol(k, 0.1) == pytest.approx(np.sqrt(s1.total_variance(k) / 0.25))
    assert surface.implied_vol(k, 3.0) == pytest.approx(np.sqrt(s2.total_variance(k) / 1.0))


def test_total_variance_is_nondecreasing_in_time_for_ordered_slices():
    slices = [(t, _slice_for(t)) for t in (0.1, 0.3, 0.6, 1.5)]
    surface = Surface(slices)
    assert surface.calendar_violations(np.linspace(-1, 1, 201)) == []
    times = np.linspace(0.02, 2.5, 120)
    w = np.array([surface.total_variance(0.2, t) for t in times])
    assert np.all(np.diff(w) >= -1e-12)


def test_calendar_violation_is_reported_for_crossing_slices():
    early = _slice_for(0.5)
    late = SviSlice(a=early.a - 0.01, b=early.b * 0.5, rho=early.rho, m=0.0, sigma=early.sigma)
    surface = Surface([(0.5, early), (1.0, late)])
    found = surface.calendar_violations(np.linspace(-1, 1, 201))
    assert len(found) == 1
    assert found[0][0] == 0.5 and found[0][1] == 1.0 and found[0][2] < 0


def test_fit_recovers_a_consistent_surface_without_noise():
    slices = [(t, _slice_for(t)) for t in (0.05, 0.25, 0.5, 1.0)]
    surface, fits = fit_surface(_quotes(slices))
    assert max(f.max_abs_error for f in fits) < 1e-6
    for t, truth in slices:
        np.testing.assert_allclose(surface.total_variance(K, t), truth.total_variance(K), atol=1e-6)
    assert surface.calendar_violations(np.linspace(-1, 1, 201)) == []


def test_repair_removes_a_calendar_violation_caused_by_noisy_quotes():
    # Two expiries quoted off the same smile: the true calendar spread is worth nothing, so
    # independent fits of noisy quotes cross each other.
    rng = np.random.default_rng(2)
    truth = _slice_for(0.25)
    noisy = [
        SliceQuotes(t, K, truth.total_variance(K) + rng.normal(0, 6e-4, K.size))
        for t in (0.25, 0.30)
    ]
    grid = np.linspace(-0.8, 0.8, 321)
    raw, _ = fit_surface(noisy, repair_calendar=False)
    fixed, _ = fit_surface(noisy, repair_calendar=True)
    assert raw.calendar_violations(grid)  # the unrepaired fit crosses
    assert fixed.calendar_violations(grid, tol=1e-6) == []


def test_unsorted_quotes_are_accepted():
    slices = [(t, _slice_for(t)) for t in (0.5, 0.1, 1.0)]
    surface, _ = fit_surface(_quotes(slices))
    np.testing.assert_allclose(surface.times, [0.1, 0.5, 1.0])


def test_fit_errors_are_reported_against_the_original_quotes():
    rng = np.random.default_rng(2)
    truth = _slice_for(0.25)
    noisy = [
        SliceQuotes(t, K, truth.total_variance(K) + rng.normal(0, 6e-4, K.size))
        for t in (0.25, 0.30)
    ]
    surface, fits = fit_surface(noisy)
    for quote, fit, fitted in zip(noisy, fits, surface.slices, strict=True):
        error = fitted.total_variance(quote.k) - quote.w
        assert fit.rmse == pytest.approx(np.sqrt(np.mean(error**2)))
        assert fit.max_abs_error == pytest.approx(np.max(np.abs(error)))


def test_a_crossing_beyond_the_quotes_is_repaired_only_on_an_explicit_wider_grid():
    early = SviSlice(a=0.01, b=0.12, rho=-0.3, m=0.0, sigma=0.1)
    late = SviSlice(a=0.04, b=0.05, rho=-0.3, m=0.0, sigma=0.15)
    k = np.linspace(-0.3, 0.3, 13)  # both slices are ordered here and cross near -0.33 and 0.6
    quotes = [
        SliceQuotes(0.25, k, early.total_variance(k)),
        SliceQuotes(0.5, k, late.total_variance(k)),
    ]
    quoted, wide = np.linspace(-0.3, 0.3, 121), np.linspace(-1.0, 1.0, 401)

    default, _ = fit_surface(quotes)
    assert default.calendar_violations(quoted, tol=1e-6) == []
    assert default.calendar_violations(wide, tol=1e-6)  # nothing constrains the wings

    repaired, _ = fit_surface(quotes, k_grid=wide)
    assert repaired.calendar_violations(wide, tol=1e-6) == []
    relative = np.abs(repaired.slices[1].total_variance(k) - late.total_variance(k))
    assert (relative / late.total_variance(k)).max() < 0.1  # paid for with a few percent


def test_expiries_that_quote_disjoint_ranges_are_fitted_without_a_repair():
    low, high = np.linspace(-0.4, -0.1, 9), np.linspace(0.1, 0.4, 9)
    quotes = [
        SliceQuotes(0.25, low, _slice_for(0.25).total_variance(low)),
        SliceQuotes(0.5, high, _slice_for(0.5).total_variance(high)),
    ]
    surface, fits = fit_surface(quotes)
    assert len(fits) == 2 and max(f.max_abs_error for f in fits) < 1e-6
