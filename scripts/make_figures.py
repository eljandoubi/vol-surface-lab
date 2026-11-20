"""Figures for the README, from the JSON in results/ and one fresh fit of a noisy surface.

python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import fit_quality as fq  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from volsurf.surface import fit_surface  # noqa: E402

RESULTS = Path("results")


def smiles() -> None:
    vols = fq.true_vols()
    rng = np.random.default_rng(0)
    noisy = [v + rng.normal(0.0, 0.003, v.shape) for v in vols]
    surface, _ = fit_surface(fq.quotes_from(noisy))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for t, truth, quotes in zip(fq.EXPIRIES, vols, noisy, strict=True):
        months = round(12 * t)
        k, label = fq.log_moneyness(t), f"{months} month" + ("s" if months > 1 else "")
        (line,) = ax.plot(fq.STANDARD_DEVIATIONS, 100 * truth, lw=1, alpha=0.5)
        colour = line.get_color()
        ax.plot(fq.STANDARD_DEVIATIONS, 100 * quotes, ".", ms=4, color=colour)
        fitted = np.sqrt(surface.total_variance(k, t) / t)
        ax.plot(fq.STANDARD_DEVIATIONS, 100 * fitted, "--", lw=1.4, color=colour, label=label)
    ax.set(xlabel="log-moneyness in standard deviations", ylabel="implied volatility (%)")
    ax.set_title("SVI fit (dashed) to noisy quotes (dots) of a Heston surface (thin line)")
    ax.legend(title="expiry")
    fig.tight_layout()
    fig.savefig(RESULTS / "smiles.png", dpi=130)


def hedging() -> None:
    data = json.loads((RESULTS / "hedging.json").read_text())
    names = ["bs_entry_vol", "bs_current_vol", "heston_delta", "heston_min_variance"]
    labels = [
        "BS delta,\nentry vol",
        "BS delta,\ncurrent vol",
        "model delta,\nfixed variance",
        "minimum-\nvariance delta",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.38
    for i, freq in enumerate(data["frequencies"]):
        rows = [freq["summary"][n] for n in names]
        ax.bar(
            np.arange(len(names)) + (i - 0.5) * width,
            [r["std_over_premium"] for r in rows],
            width,
            yerr=[1.96 * r["std_standard_error"] for r in rows],
            label="once a day"
            if freq["rebalances_per_day"] == 1
            else f"{freq['rebalances_per_day']} times a day",
        )
    ax.set_xticks(np.arange(len(names)), labels)
    ax.set(ylabel="std of profit and loss / premium")
    ax.set_title("Hedging a short 30-day call under Heston, same 10,000 paths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "hedging.png", dpi=130)


if __name__ == "__main__":
    smiles()
    hedging()
