"""Delta hedging a short at-the-money call in a Heston world with four hedge ratios.

    python scripts/hedging_experiment.py [--paths 10000] [--days 30] [--out results/hedging.json]

The hedger sells one call at its Heston price and rebalances once or four times a day. All
strategies are run on the same simulated paths, so spreads are compared with a paired bootstrap
over paths.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from volsurf import hedging
from volsurf.heston import HestonParams

PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.6, rho=-0.7)
REFERENCE = "heston_min_variance"


def one_frequency(days: int, per_day: int, paths: int, seed: int) -> dict:
    started = time.time()
    result = hedging.run_hedge(
        PARAMS, days=days, rebalances_per_day=per_day, n_paths=paths, seed=seed
    )
    summary = hedging.summarise(result)
    ratios = {}
    for name in result.pnl:
        if name == REFERENCE:
            continue
        ratio, low, high = hedging.paired_std_ratio(
            result.pnl[name], result.pnl[REFERENCE], n_boot=500
        )
        ratios[name] = {"std_over_reference": ratio, "ci95_low": low, "ci95_high": high}
    return {
        "rebalances_per_day": per_day,
        "entry_price": result.entry_price,
        "entry_implied_vol": result.entry_vol,
        "missing_implied_vol_fraction": result.missing_implied_vol_fraction,
        "summary": summary,
        f"std_ratio_to_{REFERENCE}": ratios,
        "seconds": round(time.time() - started, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paths", type=int, default=10000)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/hedging.json"))
    args = parser.parse_args()

    result = {
        "heston": asdict(PARAMS),
        "feller_margin": PARAMS.feller,
        "days": args.days,
        "paths": args.paths,
        "seed": args.seed,
        "frequencies": [one_frequency(args.days, n, args.paths, args.seed) for n in (1, 4)],
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
