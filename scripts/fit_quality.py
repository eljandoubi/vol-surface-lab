"""How well raw SVI slices fit a Heston surface, without and with quote noise.

    python scripts/fit_quality.py [--seeds 20] [--noise 0.003] [--out results/fit_quality.json]

The "market" is a Heston surface, so the true implied volatility is known at every quote. Quotes sit
at equal numbers of standard deviations for every expiry, as a strike grid in delta would give.
Slices are fitted one expiry at a time; the calendar-spread repair of ``fit_surface`` is compared
with no repair, and arbitrage is checked both inside the quoted range and far outside it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy

from volsurf.heston import HestonParams, implied_vols
from volsurf.surface import SliceQuotes, fit_surface

PARAMS = HestonParams(v0=0.04, kappa=1.5, theta=0.05, xi=0.5, rho=-0.7)
EXPIRIES = (1 / 12, 0.25, 0.5, 1.0, 2.0)
STANDARD_DEVIATIONS = np.linspace(-2.5, 2.5, 29)
LEVEL = 0.22  # rough at-the-money volatility that sets the width of the strike grid
NEAR = np.linspace(-0.9, 0.9, 361)  # covers every quote and a margin; the repair works here
WIDE = np.linspace(-3.0, 3.0, 601)  # far wings, where no quote constrains the slices


def log_moneyness(t: float) -> np.ndarray:
    """ln(K / F) of the quotes of one expiry."""
    return STANDARD_DEVIATIONS * LEVEL * np.sqrt(t)


def true_vols() -> list[np.ndarray]:
    """Heston implied vols per expiry; spot 1 and zero rates, so F = 1 and K = exp(k)."""
    return [implied_vols(1.0, np.exp(log_moneyness(t)), t, PARAMS) for t in EXPIRIES]


def quotes_from(vols) -> list[SliceQuotes]:
    return [SliceQuotes(t, log_moneyness(t), v**2 * t) for t, v in zip(EXPIRIES, vols, strict=True)]


def vol_errors(surface, vols) -> list[np.ndarray]:
    """Fitted minus true implied volatility at every quote, in volatility points."""
    return [
        100.0 * (np.sqrt(surface.total_variance(log_moneyness(t), t) / t) - v)
        for t, v in zip(EXPIRIES, vols, strict=True)
    ]


def rmse(errors) -> float:
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))


def quoted_overlap_violations(surface) -> int:
    """Consecutive pairs that cross where both expiries are quoted."""
    count = 0
    for i in range(len(EXPIRIES) - 1):
        reach = log_moneyness(EXPIRIES[i]).max()  # the earlier expiry has the narrower range
        grid = np.linspace(-reach, reach, 201)
        gap = surface.slices[i + 1].total_variance(grid) - surface.slices[i].total_variance(grid)
        count += int(gap.min() < -1e-9)
    return count


def arbitrage(surface) -> dict:
    return {
        "calendar_pairs_violated_quoted": quoted_overlap_violations(surface),
        "calendar_pairs_violated_near": len(surface.calendar_violations(NEAR, tol=1e-9)),
        "calendar_pairs_violated_wide": len(surface.calendar_violations(WIDE, tol=1e-9)),
        "expiries_with_butterfly_violation_near": len(surface.butterfly_violations(NEAR)),
        "expiries_with_butterfly_violation_wide": len(surface.butterfly_violations(WIDE)),
    }


# raw: no repair; repaired: where both expiries are quoted; repaired_wide: on the whole wide grid
VARIANTS = {
    "raw": {"repair_calendar": False},
    "repaired": {},
    "repaired_wide": {"k_grid": WIDE},
}


def clean_case(vols) -> dict:
    surface, _ = fit_surface(quotes_from(vols))
    errors = vol_errors(surface, vols)
    return {
        "rmse_vol_points_per_expiry": [rmse([e]) for e in errors],
        "max_abs_error_vol_points": float(max(np.abs(e).max() for e in errors)),
        **arbitrage(surface),
    }


def noisy_cases(vols, noise: float, seeds: int) -> dict:
    runs = {name: [] for name in VARIANTS}
    quote_noise = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        noisy = [v + rng.normal(0.0, noise, v.shape) for v in vols]
        quote_noise.append(100.0 * rmse([n - v for n, v in zip(noisy, vols, strict=True)]))
        for name, kwargs in VARIANTS.items():
            surface, _ = fit_surface(quotes_from(noisy), **kwargs)
            runs[name].append({"rmse": rmse(vol_errors(surface, vols)), **arbitrage(surface)})
    summary = {"quote_noise_rmse_vol_points": float(np.mean(quote_noise)), "seeds": seeds}
    for name, rows in runs.items():
        errors = np.array([row["rmse"] for row in rows])
        summary[name] = {
            "rmse_vs_truth_vol_points_mean": float(errors.mean()),
            "rmse_vs_truth_vol_points_worst_run": float(errors.max()),
            **{
                f"runs_with_{key}": int(sum(row[key] > 0 for row in rows))
                for key in rows[0]
                if key != "rmse"
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--noise", type=float, default=0.003, help="std of the vol noise per quote")
    parser.add_argument("--out", type=Path, default=Path("results/fit_quality.json"))
    args = parser.parse_args()

    started = time.time()
    vols = true_vols()
    result = {
        "heston": vars(PARAMS),
        "expiries": list(EXPIRIES),
        "quotes_per_expiry": len(STANDARD_DEVIATIONS),
        "quote_range_standard_deviations": [
            float(STANDARD_DEVIATIONS[0]),
            float(STANDARD_DEVIATIONS[-1]),
        ],
        "noise_vol_points": 100.0 * args.noise,
        "clean": clean_case(vols),
        "noisy": noisy_cases(vols, args.noise, args.seeds),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "seconds": round(time.time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
