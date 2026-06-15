#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_may_2024_event.py

Simple two-line plot of the May 10 2024 NZ supply shortfall event.
Shows total generation vs total demand -- the razor-thin margin tells
the story clearly without any confusing stacked area complexity.

Output:
    outputs/figures/appendix/may_2024_generation_mix.png
"""
import os, sys, logging, matplotlib
from pathlib import Path
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

root = os.environ.get('PYPSA_NZA_ROOT', '')
if not root:
    log.error("PYPSA_NZA_ROOT not set"); sys.exit(1)

workspace   = Path(root) / 'pypsa_nza_workspace'
annual_dir  = workspace / 'data' / 'processed' / 'annual'
fig_dir     = workspace / 'outputs' / 'figures' / 'appendix'
fig_dir.mkdir(parents=True, exist_ok=True)

EVENT_DATE = '2024-05-10'
C_GEN      = '#7aadbe'   # muted steel blue -- generation
C_DEM      = '#111111'   # near black -- demand
C_MARGIN   = '#d4a860'   # muted amber -- reserve margin fill
C_EVENT    = '#a63b2b'   # muted brick red -- event marker
C_RESPONSE = '#2e7d32'   # dark green -- demand response

def load_day(filepath, date_str):
    df = pd.read_csv(str(filepath), index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df[df.index.date == pd.Timestamp(date_str).date()]

def plot_event():
    gen_raw  = load_day(annual_dir / '2024_import_sites_all.csv', EVENT_DATE)
    dem_raw  = load_day(annual_dir / '2024_export_sites_all.csv', EVENT_DATE)

    total_gen = gen_raw.sum(axis=1)
    total_dem = dem_raw.sum(axis=1)
    reserve   = total_gen - total_dem
    times     = total_gen.index

    t_notice   = pd.Timestamp('2024-05-10 17:00:00')
    t_response = pd.Timestamp('2024-05-10 17:30:00')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8),
                                    gridspec_kw={'height_ratios': [2.5, 1]},
                                    sharex=True)
    fig.subplots_adjust(hspace=0.06)

    # -------------------------------------------------------------------
    # TOP PANEL: Total generation vs total demand
    # -------------------------------------------------------------------
    # Shade the margin between generation and demand
    ax1.fill_between(times, total_dem.values, total_gen.values,
                     color=C_MARGIN, alpha=0.5, label='Reserve margin')

    ax1.plot(times, total_gen.values, color=C_GEN, linewidth=2.5,
             label='Total generation (GIP)', zorder=4)
    ax1.plot(times, total_dem.values, color=C_DEM, linewidth=2.5,
             label='Total demand (GXP)', zorder=5)

    # Event markers
    ax1.axvline(t_notice,   color=C_EVENT,    linewidth=2.0,
                linestyle='--', zorder=6)
    ax1.axvline(t_response, color=C_RESPONSE, linewidth=1.5,
                linestyle=':', zorder=6)

    ax1.annotate('Conservation\nnotice ~17:00',
                 xy=(t_notice, 3000),
                 xytext=(pd.Timestamp('2024-05-10 13:30'), 3080),
                 fontsize=9, fontweight='bold', color=C_EVENT,
                 arrowprops=dict(arrowstyle='->', color=C_EVENT, lw=1.3),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=C_EVENT, alpha=0.95), zorder=8)

    ax1.annotate('Demand response\n~260 MW reduction',
                 xy=(t_response, 2900),
                 xytext=(pd.Timestamp('2024-05-10 19:30'), 3080),
                 fontsize=9, fontweight='bold', color=C_RESPONSE,
                 arrowprops=dict(arrowstyle='->', color=C_RESPONSE, lw=1.3),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=C_RESPONSE, alpha=0.95), zorder=8)

    ax1.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax1.set_ylabel('Power (MW)', fontsize=11)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(250))
    ax1.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_title(
        'New Zealand Electricity System -- 10 May 2024 Supply Shortfall Event\n'
        'Total generation vs total demand: reserve margin below 112~MW '
        '(< 4\\% of peak demand) throughout the day',
        fontsize=11, fontweight='bold')

    # -------------------------------------------------------------------
    # BOTTOM PANEL: Reserve margin
    # -------------------------------------------------------------------
    ax2.fill_between(times, 0, reserve.values,
                     color=C_MARGIN, alpha=0.7, label='Reserve margin')
    ax2.plot(times, reserve.values, color='#8b6914', linewidth=1.2)
    ax2.axhline(0,   color=C_DEM,   linewidth=1.0)
    ax2.axhline(260, color=C_EVENT, linewidth=1.0, linestyle=':',
                alpha=0.6)
    ax2.text(pd.Timestamp('2024-05-10 00:15'), 265,
             '260 MW demand response', color=C_EVENT, fontsize=8,
             va='bottom', fontweight='bold')

    ax2.axvline(t_notice,   color=C_EVENT,    linewidth=2.0,
                linestyle='--', zorder=5)
    ax2.axvline(t_response, color=C_RESPONSE, linewidth=1.5,
                linestyle=':', zorder=5)

    # Annotate minimum
    min_idx = reserve.idxmin()
    min_val = reserve.min()
    ax2.annotate(f'Min: {min_val:.0f} MW\n{min_idx.strftime("%H:%M")}',
                 xy=(min_idx, min_val),
                 xytext=(pd.Timestamp('2024-05-10 05:00'), 20),
                 fontsize=8, color='#8b6914', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#8b6914', lw=1.0),
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='#8b6914', alpha=0.9))

    ax2.set_ylabel('Reserve margin\n(MW)', fontsize=10)
    ax2.set_ylim(-50, 350)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.legend(fontsize=9, loc='upper right', framealpha=0.95)

    ax2.set_xlabel('Time of day (10 May 2024)', fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    fig.tight_layout()
    out = fig_dir / 'may_2024_generation_mix.png'
    fig.savefig(str(out), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("Saved: %s", out)

if __name__ == '__main__':
    plot_event()
