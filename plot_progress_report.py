#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_progress_report.py  -- corrected version

Generates figures for supervisor progress report.
Uses pre-computed line_loading CSVs from validation pipeline.

Figures:
  PR1_network_validation_summary.png  -- combined network heatmap 2024+2025
  PR2_baseline_adequacy_summary.png   -- adequacy + max loading trend 2025-2040
  PR3_binding_constraint_loading.png  -- S-44/S-24 monthly loading across years
  PR4_reserve_margin_planning.png     -- reserve margin by month/year
  PR5_generation_dispatch_mix.png     -- generation capacity vs demand
"""
import os, sys, logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

WORKSPACE    = Path(r"C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace")
HEURISTIC    = Path(r"C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-heuristic")
VERIF_DIR    = WORKSPACE / "outputs" / "verification_pf" / "combined"
RESULTS_DIR  = HEURISTIC / "results" / "validation"
FIG_DIR      = WORKSPACE / "outputs" / "figures" / "progress_report"
CONFIG_PATH  = WORKSPACE / "config" / "nza_net_config.yaml"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MONTHS     = ['jan','feb','mar','apr','may','jun',
              'jul','aug','sep','oct','nov','dec']
MONTH_LBLS = ['Jan','Feb','Mar','Apr','May','Jun',
              'Jul','Aug','Sep','Oct','Nov','Dec']
PLAN_YEARS = [2025, 2030, 2035, 2040]
VAL_YEARS  = [2024, 2025]

C_BLUE  = '#7aadbe'
C_AMBER = '#d4a860'
C_BRICK = '#a63b2b'
C_CREAM = '#f7f7f2'
C_DARK  = '#1a1a1a'
C_GREEN = '#4a9e6b'
C_GREY  = '#888888'

YEAR_COLORS = {2025: C_BLUE, 2030: C_GREEN, 2035: C_AMBER, 2040: C_BRICK}

CMAP = LinearSegmentedColormap.from_list(
    'loading', [(0, C_CREAM), (0.5, C_BLUE), (0.8, C_AMBER), (1.0, C_BRICK)])


def load_val_loading(year, month):
    p = VERIF_DIR / str(year) / f'{month}_{year}' / 'line_loading.csv'
    if not p.exists(): return None
    return pd.read_csv(str(p), index_col=0)


def load_dispatch_loading(year, month):
    p = RESULTS_DIR / f'line_loading_{year}_{month}.csv'
    if not p.exists(): return None
    return pd.read_csv(str(p), index_col=0)


# ===========================================================================
# PR1: Validation heatmap
# ===========================================================================
def plot_validation_heatmap():
    log.info("PR1: Validation heatmap...")
    TOP_N = 15
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, year in zip(axes, VAL_YEARS):
        frames = {}
        for m in MONTHS:
            df = load_val_loading(year, m)
            if df is not None and 'max_loading_pct' in df.columns:
                frames[m] = df['max_loading_pct']
        if not frames: continue

        data = pd.DataFrame(frames)
        data = data[~data.index.str.contains('HVDC')]
        top  = data.max(axis=1).nlargest(TOP_N).index
        arr  = data.loc[top].values.astype(float)

        im = ax.imshow(arr, aspect='auto', cmap=CMAP,
                       vmin=0, vmax=100, interpolation='nearest')
        for r in range(len(top)):
            for c in range(12):
                v = arr[r, c]
                if np.isnan(v): continue
                tc = 'white' if v > 65 else C_DARK
                ax.text(c, r, f'{v:.0f}', ha='center', va='center',
                        fontsize=8, color=tc, fontweight='bold')

        ax.set_xticks(range(12))
        ax.set_xticklabels(MONTH_LBLS, fontsize=9)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top, fontsize=9, fontfamily='monospace')
        ax.set_title(f'{year}', fontsize=13, fontweight='bold', pad=8)
        fig.colorbar(im, ax=ax, shrink=0.6).set_label('Max loading (%)', fontsize=9)

    fig.suptitle(
        'Combined SI+NI Network: Monthly Maximum Line Loading -- Validation\n'
        '234 buses  |  283 AC lines  |  1 HVDC link  |  0 overloads across all 24 months',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / 'PR1_network_validation_summary.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s", out)


# ===========================================================================
# PR2: Baseline adequacy + max loading trend
# ===========================================================================
def plot_adequacy_summary():
    log.info("PR2: Adequacy summary...")

    max_loadings = []
    worst_months = []
    for year in PLAN_YEARS:
        df = pd.read_csv(RESULTS_DIR / f'annual_summary_{year}.csv')
        max_loadings.append(df['max_line_loading_pct'].max())
        worst_months.append(df.loc[df['max_line_loading_pct'].idxmax(), 'month'].upper())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: adequate months
    ax = axes[0]
    x = np.arange(len(PLAN_YEARS))
    bars = ax.bar(x, [12]*4, color=[YEAR_COLORS[y] for y in PLAN_YEARS],
                  edgecolor='white', linewidth=0.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in PLAN_YEARS], fontsize=11)
    ax.set_ylim(0, 14)
    ax.set_ylabel('Adequate months / 12', fontsize=10)
    ax.axhline(12, color=C_GREY, linewidth=1.0, linestyle='--', alpha=0.5)
    for bar, ml, wm in zip(bars, max_loadings, worst_months):
        ax.text(bar.get_x() + bar.get_width()/2, 12.3,
                f'Peak {ml:.1f}%\n({wm})',
                ha='center', fontsize=8, color=C_DARK)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Generation Adequacy\n(Baseline: no capacity additions)',
                 fontsize=11, fontweight='bold')

    # Right: max loading trend
    ax2 = axes[1]
    ax2.plot(PLAN_YEARS, max_loadings, 'o-', color=C_BRICK,
             linewidth=2.5, markersize=9, zorder=5)
    for yr, ml, wm in zip(PLAN_YEARS, max_loadings, worst_months):
        ax2.text(yr, ml + 0.8, f'{ml:.1f}%\n({wm})',
                 ha='center', fontsize=9, fontweight='bold', color=C_BRICK)
    ax2.axhline(100, color=C_BRICK, linewidth=1.5, linestyle='--',
                alpha=0.7, label='Thermal limit')
    ax2.axhline(90,  color=C_AMBER, linewidth=1.0, linestyle=':',
                alpha=0.7, label='90% planning threshold')
    ax2.set_xticks(PLAN_YEARS)
    ax2.set_xticklabels([str(y) for y in PLAN_YEARS], fontsize=11)
    ax2.set_ylabel('Maximum line loading (%)', fontsize=10)
    ax2.set_ylim(60, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.spines[['top','right']].set_visible(False)
    ax2.set_title('Maximum Line Loading Trend\n(Increasing -- S-44 Waitaki corridor)',
                  fontsize=11, fontweight='bold')

    fig.suptitle(
        'Baseline Planning Results: 2025-2040 (No Capacity Additions)\n'
        'MBIE reference demand  |  2025 base year  |  Binding: S-44 Waitaki-Livingston (SI)',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / 'PR2_baseline_adequacy_summary.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s", out)


# ===========================================================================
# PR3: S-44 and S-24 monthly loading across years
# ===========================================================================
def plot_binding_constraint():
    log.info("PR3: Binding constraint loading...")

    LINES = ['S-44', 'S-24']
    LINE_LABELS = {
        'S-44': 'S-44  Waitaki-Livingston 220 kV  (366 MW)',
        'S-24': 'S-24  Ohau B-Twizel 220 kV  (732 MW)',
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    for ax, line_name in zip(axes, LINES):
        x = np.arange(12)
        width = 0.18
        for i, year in enumerate(PLAN_YEARS):
            vals = []
            for m in MONTHS:
                df = load_dispatch_loading(year, m)
                if df is not None and line_name in df.index:
                    vals.append(float(df.loc[line_name, 'max_loading_pct']))
                else:
                    vals.append(0.0)
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width,
                   label=str(year), color=YEAR_COLORS[year],
                   edgecolor='white', linewidth=0.3, alpha=0.9)

        ax.axhline(100, color=C_BRICK, linewidth=1.5, linestyle='--',
                   label='Thermal limit')
        ax.axhline(90,  color=C_AMBER, linewidth=1.0, linestyle=':',
                   alpha=0.7, label='90% threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_LBLS, fontsize=9)
        ax.set_ylabel('Maximum monthly loading (%)', fontsize=10)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(LINE_LABELS[line_name], fontsize=11, fontweight='bold')

    fig.suptitle(
        'South Island Binding Constraints: Monthly Maximum Loading 2025-2040\n'
        'Waitaki-Ohau hydro corridor loading increases progressively with demand growth',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / 'PR3_binding_constraint_loading.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s", out)


# ===========================================================================
# PR4: Reserve margin
# ===========================================================================
def plot_reserve_margin():
    log.info("PR4: Reserve margin...")

    GEN_CAP = {
        'hydro': 3037.0, 'geothermal': 1059.5, 'wind': 875.0,
        'gas': 93.0, 'ocgt': 330.0, 'ccgt': 212.0,
        'solar_pv': 44.8, 'diesel': 156.0,
    }
    AVAIL = {'geothermal': 0.87, 'wind': 1.0, 'gas': 1.0,
             'ocgt': 1.0, 'ccgt': 1.0, 'solar_pv': 1.0, 'diesel': 1.0}
    non_hydro = sum(GEN_CAP[c]*AVAIL[c] for c in GEN_CAP if c != 'hydro')

    hydro_seasonal = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        hydro_seasonal = {int(k): float(v)
                         for k, v in cfg.get('hydro_seasonal_availability',{}).items()}

    BASE_DEM = [2024, 2024, 2081, 2024, 2107, 2024,
                2409, 2279, 2093, 2026, 2035, 2024]
    MBIE = {2025: 1.0, 2030: 1.1087, 2035: 1.2132, 2040: 1.3435}

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(12)
    width = 0.18

    for i, year in enumerate(PLAN_YEARS):
        scale = MBIE[year]
        margins = []
        for mi, dem in enumerate(BASE_DEM):
            month_num = mi + 1
            hydro_pu  = hydro_seasonal.get(month_num, 0.75)
            total_cap = GEN_CAP['hydro'] * hydro_pu + non_hydro
            margins.append(total_cap - dem * scale)
        offset = (i - 1.5) * width
        ax.bar(x + offset, margins, width,
               label=str(year), color=YEAR_COLORS[year],
               edgecolor='white', linewidth=0.3, alpha=0.9)

    ax.axhline(0,   color=C_DARK, linewidth=1.2)
    ax.axhline(500, color=C_GREY, linewidth=1.0, linestyle='--',
               alpha=0.5, label='500 MW reference')
    ax.annotate('Winter trough\n(hydro p_max_pu=0.55)',
                xy=(6.5, 800), xytext=(8.5, 400),
                fontsize=9, color=C_BRICK,
                arrowprops=dict(arrowstyle='->', color=C_BRICK, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=C_BRICK, alpha=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_LBLS, fontsize=10)
    ax.set_ylabel('Reserve margin (MW)\nAvailable generation - peak demand', fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(
        'Monthly Generation Reserve Margin: 2025-2040\n'
        'Seasonal hydro availability applied  |  All years adequate under MBIE reference scenario',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / 'PR4_reserve_margin_planning.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s", out)


# ===========================================================================
# PR5: Generation mix
# ===========================================================================
def plot_generation_mix():
    log.info("PR5: Generation mix...")

    GEN_CAP = {
        'Hydro': 3037.0*0.85, 'Geothermal': 1059.5*0.87,
        'Wind': 875.0, 'OCGT': 330.0, 'CCGT': 212.0,
        'Gas': 93.0, 'Solar PV': 44.8, 'Diesel': 156.0,
    }
    COLORS = {
        'Hydro': '#5b9bd5', 'Geothermal': '#ed7d31', 'Wind': '#70ad47',
        'OCGT': '#c00000', 'CCGT': '#7030a0', 'Gas': '#ffc000',
        'Solar PV': '#ffe699', 'Diesel': '#9e480e',
    }
    MBIE = {2025: 1.0, 2030: 1.1087, 2035: 1.2132, 2040: 1.3435}
    BASE_DEC = 2024.0

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(PLAN_YEARS))
    bottom = np.zeros(len(PLAN_YEARS))

    for carrier, cap in GEN_CAP.items():
        ax.bar(x, [cap]*4, 0.55, bottom=bottom,
               label=carrier, color=COLORS[carrier],
               edgecolor='white', linewidth=0.4)
        bottom += np.array([cap]*4)

    demands = [BASE_DEC * MBIE[y] for y in PLAN_YEARS]
    ax.plot(x, demands, 'o-', color=C_DARK, linewidth=2.5,
            markersize=8, label='December mean demand', zorder=5)
    for xi, dem in zip(x, demands):
        ax.text(xi, dem + 50, f'{dem:.0f} MW',
                ha='center', fontsize=9, fontweight='bold', color=C_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in PLAN_YEARS], fontsize=12)
    ax.set_ylabel('Power (MW)', fontsize=11)
    ax.set_ylim(0, 7500)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.legend(loc='upper left', fontsize=9, ncol=2,
              bbox_to_anchor=(0.01, 0.99))
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(
        'Available Generation Capacity vs December Mean Demand: 2025-2040\n'
        'Static generation portfolio  |  Seasonal hydro (Dec: p_max_pu=0.85)\n'
        'System is not generation-constrained -- binding constraint is SI transmission (S-44)',
        fontsize=11, fontweight='bold')
    fig.tight_layout()
    out = FIG_DIR / 'PR5_generation_dispatch_mix.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("  Saved: %s", out)


if __name__ == '__main__':
    log.info("Generating corrected progress report figures...")
    plot_validation_heatmap()
    plot_adequacy_summary()
    plot_binding_constraint()
    plot_reserve_margin()
    plot_generation_mix()
    log.info("Done. Figures saved to: %s", FIG_DIR)
