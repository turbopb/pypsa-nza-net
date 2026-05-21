#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ni_line_loading.py

NI line loading comparison bar chart for thesis appendix.
Mirrors plot_dc_validation.py Figure 2 for the North Island.

Generates:
    ni_line_loading_comparison.png  -- Jan vs Jul 2024, top 20 lines

Usage
-----
    python plot_ni_line_loading.py

Output written to:
    {workspace}/outputs/figures/validation/ni_line_loading_comparison.png
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

_root = os.environ.get("PYPSA_NZA_ROOT", "")
if not _root:
    log.error("PYPSA_NZA_ROOT not set")
    sys.exit(1)

workspace = Path(_root) / "pypsa_nza_workspace"
verif_dir = workspace / "outputs" / "verification_pf" / "NI"
fig_dir   = workspace / "outputs" / "figures" / "validation"
fig_dir.mkdir(parents=True, exist_ok=True)

MONTHS = ["jan","feb","mar","apr","may","jun",
          "jul","aug","sep","oct","nov","dec"]

def load_loading(year, month):
    month_label = f"{MONTHS[month-1]}_{year}"
    path = verif_dir / str(year) / month_label / "line_loading.csv"
    if not path.exists():
        log.warning("Not found: %s", path)
        return None
    return pd.read_csv(str(path), index_col=0)


def plot_ni_loading_comparison():
    log.info("NI line loading comparison: Jan vs Jul 2024")

    jan = load_loading(2024, 1)
    jul = load_loading(2024, 7)

    if jan is None or jul is None:
        log.error("Missing loading data -- run nza_net_pf.py --island NI first")
        return

    # Top 20 lines by max loading in either month
    all_lines = sorted(set(jan.index) | set(jul.index))
    max_either = []
    for ln in all_lines:
        j = float(jan.loc[ln, "max_loading_pct"]) if ln in jan.index else 0
        u = float(jul.loc[ln, "max_loading_pct"]) if ln in jul.index else 0
        max_either.append(max(j, u))

    top_idx   = sorted(range(len(all_lines)),
                       key=lambda i: max_either[i], reverse=True)[:20]
    top_lines = [all_lines[i] for i in top_idx]

    jan_mean = [float(jan.loc[ln, "mean_loading_pct"])
                if ln in jan.index else 0 for ln in top_lines]
    jan_max  = [float(jan.loc[ln, "max_loading_pct"])
                if ln in jan.index else 0 for ln in top_lines]
    jul_mean = [float(jul.loc[ln, "mean_loading_pct"])
                if ln in jul.index else 0 for ln in top_lines]
    jul_max  = [float(jul.loc[ln, "max_loading_pct"])
                if ln in jul.index else 0 for ln in top_lines]

    y   = np.arange(len(top_lines))
    h   = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                             facecolor="white", sharey=True)

    # Greyscale to match thesis colour scheme
    colours = {"jan_mean": "#aaaaaa", "jan_max": "#333333",
               "jul_mean": "#cccccc", "jul_max": "#555555"}

    for ax, mean_vals, max_vals, label, c_mean, c_max in [
        (axes[0], jan_mean, jan_max, "January 2024",
         colours["jan_mean"], colours["jan_max"]),
        (axes[1], jul_mean, jul_max, "July 2024",
         colours["jul_mean"], colours["jul_max"]),
    ]:
        ax.barh(y + h/2, max_vals,  h, color=c_max,
                label="Maximum", alpha=0.9)
        ax.barh(y - h/2, mean_vals, h, color=c_mean,
                label="Mean", alpha=0.9)
        ax.axvline(100, color="#111111", linestyle="--",
                   linewidth=1.2, label="Rated capacity")
        ax.axvline(75,  color="#555555", linestyle=":",
                   linewidth=1.0, label="75% threshold")
        ax.set_xlabel("Loading (% of rated capacity)", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, max(max(jan_max + jul_max) * 1.1, 105))
        ax.tick_params(labelsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(top_lines, fontsize=8)

    fig.suptitle(
        "North Island Transmission Line Loading -- DC Power Flow Validation\n"
        "Top 20 lines by maximum loading, 2024",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out = fig_dir / "ni_line_loading_comparison.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)
    return out


if __name__ == "__main__":
    plot_ni_loading_comparison()
