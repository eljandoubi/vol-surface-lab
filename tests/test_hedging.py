import numpy as np
import pytest

from volsurf import hedging
from volsurf.heston import HestonParams

PARAMS = HestonParams(v0=0.09, kappa=2.0, theta=0.09, xi=0.6, rho=-0.5)


@pytest.fixture(scope="module")
def result():
    return hedging.run_hedge(PARAMS, days=6, rebalances_per_day=1, n_paths=4000, seed=11)


def test_every_strategy_has_one_pnl_per_path(result):
    assert set(result.pnl) == set(hedging.STRATEGIES)
    assert all(v.shape == (4000,) for v in result.pnl.values())


def test_pnl_is_centred_on_zero_in_the_risk_neutral_world(result):
    summary = hedging.summarise(result)
    for name, row in summary.items():
        assert abs(row["mean_over_premium"]) < 4 * row["mean_standard_error"] + 0.01, name


def test_hedging_reduces_the_spread_of_the_profit_and_loss(result):
    summary = hedging.summarise(result)
    unhedged = summary["none"]["std_over_premium"]
    for name in hedging.STRATEGIES[1:]:
        assert summary[name]["std_over_premium"] < 0.7 * unhedged, name


def test_entry_quantities_are_consistent(result):
    assert 0 < result.entry_price < 100
    assert 0.2 < result.entry_vol < 0.4  # sqrt(0.09) = 0.3
    assert 0 <= result.missing_implied_vol_fraction < 0.5


def test_hedge_ratios_are_bounded_and_ordered_by_moneyness():
    spot = np.array([90.0, 100.0, 110.0])
    variance = np.full(3, 0.09)
    ratios, missing = hedging.hedge_ratios(PARAMS, spot, variance, 100.0, 0.05, 0.3)
    for name, delta in ratios.items():
        assert np.all(np.diff(delta) > 0), name
        assert np.all((-0.05 < delta) & (delta < 1.05)), name
    assert 0 <= missing <= 1


def test_negative_correlation_lowers_the_minimum_variance_delta():
    spot = np.array([100.0])
    variance = np.array([0.09])
    ratios, _ = hedging.hedge_ratios(PARAMS, spot, variance, 100.0, 0.1, 0.3)
    assert ratios["heston_min_variance"][0] < ratios["heston_delta"][0]


def test_summary_standard_errors_scale_with_the_number_of_paths():
    small = hedging.run_hedge(PARAMS, days=3, n_paths=500, seed=1)
    large = hedging.run_hedge(PARAMS, days=3, n_paths=2000, seed=1)
    se_small = hedging.summarise(small)["heston_delta"]["mean_standard_error"]
    se_large = hedging.summarise(large)["heston_delta"]["mean_standard_error"]
    assert se_large < se_small


def test_paired_std_ratio_covers_the_true_ratio_and_is_exact_for_identical_samples():
    rng = np.random.default_rng(5)
    base = rng.normal(size=6000)
    noise = rng.normal(size=6000)
    a, b = 1.2 * base, base
    ratio, lo, hi = hedging.paired_std_ratio(a, b, n_boot=300)
    assert ratio == pytest.approx(1.2, rel=1e-9)
    assert lo <= 1.2 <= hi and hi - lo < 1e-6  # perfectly correlated: no sampling spread
    ratio, lo, hi = hedging.paired_std_ratio(base + 0.5 * noise, base, n_boot=400)
    expected = np.sqrt(1.25)
    assert lo < expected < hi and hi - lo < 0.05
    same = hedging.paired_std_ratio(base, base, n_boot=50)
    assert same == (1.0, 1.0, 1.0)


def test_paired_std_ratio_validates_input():
    with pytest.raises(ValueError):
        hedging.paired_std_ratio(np.ones(5), np.ones(6))
