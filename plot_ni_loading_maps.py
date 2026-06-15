#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_ni_loading_maps.py

2x4 panel NI network loading maps for December (peak N-50 month) across
planning years 2025, 2030, 2035, 2040.

Top row:    Mean half-hourly loading for December
Bottom row: Maximum half-hourly loading for December

Shows N-50 (Wellsford-Marsden) hitting 96.2% on the maximum row,
while the mean row shows typical operation. MPE (Marsden Point) is
labelled explicitly to show proximity to the saturated corridor.

Run from pypsa-nza-net directory:
    python plot_ni_loading_maps.py
"""
import os, sys, logging, yaml, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pypsa

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve root path from dispatch config -- does not require PYPSA_NZA_ROOT
# ---------------------------------------------------------------------------
def find_root():
    """Find workspace root from dispatch_config.yaml."""
    # Search upward from script location for dispatch_config.yaml
    search_dirs = [
        Path(__file__).parent,
        Path(__file__).parent.parent,
        Path(r'C:\Users\Public\Documents\Thesis\analysis'),
    ]
    for base in search_dirs:
        for cfg in base.rglob('dispatch_config.yaml'):
            try:
                with open(cfg) as f:
                    c = yaml.safe_load(f)
                root = Path(c['paths']['root']).parent
                if root.exists():
                    log.info("Root resolved from %s: %s", cfg, root)
                    return root, cfg
            except Exception:
                continue
    # Fallback to env var
    root_env = os.environ.get('PYPSA_NZA_ROOT', '')
    if root_env and Path(root_env).exists():
        cfg = Path(root_env) / 'pypsa-nza-dispatch' / 'config' / 'dispatch_config.yaml'
        return Path(root_env), cfg
    log.error("Cannot resolve root path. Set PYPSA_NZA_ROOT or ensure "
              "dispatch_config.yaml is accessible.")
    sys.exit(1)

root, config_path = find_root()
workspace    = root / 'pypsa_nza_workspace'
dispatch_dir = root / 'dispatch_data' / 'cases' / 'reference_v2'
fig_dir      = workspace / 'outputs' / 'figures' / 'progress_report'
fig_dir.mkdir(parents=True, exist_ok=True)

PLAN_YEARS = [2025, 2030, 2035, 2040]
MONTH      = 'dec'   # December -- N-50 peaks at 96.2% this month

NI_EXTENT  = [172.4, 178.6, -41.6, -34.4]

C_CREAM = '#f7f7f2'
C_BLUE  = '#7aadbe'
C_AMBER = '#d4a860'
C_BRICK = '#a63b2b'
C_DARK  = '#1a1a1a'
C_N50   = '#c0392b'
C_MPE   = '#1a5276'
C_GEO   = '#1e8449'

CMAP = mcolors.LinearSegmentedColormap.from_list(
    'loading', [(0, C_CREAM), (0.40, C_BLUE), (0.75, C_AMBER), (1.0, C_BRICK)])
NORM = mcolors.Normalize(vmin=0, vmax=100)
PROJ = ccrs.PlateCarree() if HAS_CARTOPY else None


def load_and_dispatch(year, month):
    """Load combined dispatch network and run optimisation."""
    net_dir = dispatch_dir / str(year) / f'{month}_{year}'
    if not net_dir.exists():
        log.warning("Not found: %s", net_dir); return None, None, None

    log.info("  Loading %s_%d ...", month, year)
    n = pypsa.Network()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n.import_from_csv_folder(str(net_dir))

    # Fix extendable generators
    n.generators.loc[n.generators.p_nom_extendable, 'p_nom_extendable'] = False
    n.generators.loc[n.generators.p_nom_extendable == False, 'p_nom'] = \
        n.generators.loc[n.generators.p_nom_extendable == False, 'p_nom_opt'].fillna(
        n.generators.loc[n.generators.p_nom_extendable == False, 'p_nom'])

    log.info("  Running dispatch optimisation (%d snapshots)...",
             len(n.snapshots))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        status = n.optimize(solver_name='highs',
                           solver_options={'log_to_console': False})

    if n.lines_t.p0.empty:
        log.warning("  No line flows -- optimisation may have failed")
        return n, None, None

    s_nom = n.lines.s_nom.copy()
    s_nom[s_nom == 0] = 1
    loading_pct = n.lines_t.p0.abs() / s_nom * 100

    mean_loading = loading_pct.mean()
    max_loading  = loading_pct.max()

    log.info("  N-50 mean=%.1f%%  max=%.1f%%",
             mean_loading.get('N-50', 0), max_loading.get('N-50', 0))
    return n, mean_loading, max_loading


def get_ni_elements(n):
    """Return NI buses and lines (lat > -42)."""
    buses = n.buses[(n.buses.y > -42.0) & (n.buses.x > 172.0)].copy()
    ni_idx = set(buses.index)
    lines = n.lines[n.lines.bus0.isin(ni_idx) & n.lines.bus1.isin(ni_idx)].copy()
    return buses, lines


def make_ax(fig, row, col, ncols):
    pos = row * ncols + col + 1
    if HAS_CARTOPY:
        ax = fig.add_subplot(2, ncols, pos, projection=PROJ)
        ax.set_extent(NI_EXTENT, crs=PROJ)
        ax.add_feature(cfeature.LAND,
                       facecolor='#f0ede8', zorder=0)
        ax.add_feature(cfeature.OCEAN,
                       facecolor='#daeaf5', zorder=0)
        ax.add_feature(cfeature.COASTLINE,
                       linewidth=0.6, edgecolor='#555555', zorder=1)
        ax.add_feature(cfeature.LAKES,
                       facecolor='#daeaf5', edgecolor='#aaaaaa',
                       linewidth=0.3, zorder=1)
    else:
        ax = fig.add_subplot(2, ncols, pos)
        ax.set_xlim(NI_EXTENT[0], NI_EXTENT[1])
        ax.set_ylim(NI_EXTENT[2], NI_EXTENT[3])
        ax.set_facecolor('#daeaf5')
    return ax


def draw_map(ax, n, loading, ni_buses, ni_lines, year, row_label):
    """Draw network on a single axis."""
    transform = PROJ if HAS_CARTOPY else ax.transData

    # --- Lines ---
    for line_name, line in ni_lines.iterrows():
        if line.bus0 not in n.buses.index or line.bus1 not in n.buses.index:
            continue
        b0 = n.buses.loc[line.bus0]
        b1 = n.buses.loc[line.bus1]
        pct = loading.get(line_name, 0)
        lw  = 2.5 if b0.v_nom >= 220 else 1.4

        if line_name == 'N-50':
            # Halo
            ax.plot([b0.x, b1.x], [b0.y, b1.y],
                    color=C_N50, linewidth=lw+4, alpha=0.3,
                    transform=transform, zorder=4,
                    solid_capstyle='round')
            ax.plot([b0.x, b1.x], [b0.y, b1.y],
                    color=C_N50, linewidth=lw+1,
                    transform=transform, zorder=5,
                    solid_capstyle='round')
            # Loading label
            mx = (b0.x + b1.x) / 2 + 0.18
            my = (b0.y + b1.y) / 2 + 0.05
            ax.text(mx, my, f'N-50\n{pct:.0f}%',
                    fontsize=8, fontweight='bold', color=C_N50,
                    ha='left', va='center', transform=transform, zorder=10,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor=C_N50, alpha=0.92, linewidth=1.0))
        else:
            ax.plot([b0.x, b1.x], [b0.y, b1.y],
                    color=CMAP(NORM(pct)), linewidth=lw,
                    transform=transform, zorder=3,
                    solid_capstyle='round')

    # --- Buses ---
    for bus_name, bus in ni_buses.iterrows():
        size  = 5 if bus.v_nom >= 220 else 3
        color = '#333333' if bus.v_nom >= 220 else '#777777'
        ax.plot(bus.x, bus.y, 'o', markersize=size, color=color,
                transform=transform, zorder=6, markeredgewidth=0)

    # --- Key site labels ---
    sites = [
        ('MDN_220', 'Marsden Point\n(MDN_220, electro-\nlyser site, 220kV)', C_MPE,   0.14, 0.0,  '*', 13),
        ('KOE',     'Ngāwhā\ngeothermal',                 C_GEO,   0.12, 0.0,  's', 8),
        ('HAY_220', 'Haywards\n(HVDC)',                   C_DARK,  0.12, 0.0,  'D', 7),
        ('OTA_220', 'Auckland',                           C_DARK, -0.18, 0.0,  'o', 0),
    ]
    for site, label, color, dx, dy, marker, msize in sites:
        if site not in ni_buses.index:
            # Try alternatives
            alts = {'OTA_220': ['OTA_220', 'AKL', 'HEN', 'OTA'],
                    'HAY_220': ['HAY_220', 'HAY']}
            site = next((a for a in alts.get(site, []) if a in ni_buses.index), None)
            if site is None:
                continue
        bus = ni_buses.loc[site]
        if msize > 0:
            ax.plot(bus.x, bus.y, marker, markersize=msize, color=color,
                    transform=transform, zorder=9,
                    markeredgecolor='white', markeredgewidth=0.5)
        ax.text(bus.x + dx, bus.y + dy, label,
                fontsize=7.5, color=color, fontweight='bold',
                ha='left', va='center', transform=transform, zorder=11,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=color, alpha=0.88, linewidth=0.8))

    # --- Title ---
    n50_val = loading.get('N-50', 0)
    title_color = C_BRICK if n50_val > 90 else (C_AMBER if n50_val > 70 else C_DARK)
    ax.set_title(f'{year}  |  N-50: {n50_val:.1f}%',
                 fontsize=11, fontweight='bold', color=title_color, pad=5)

    # --- Row label (left axis only) ---
    if True:  # always add row label inside map
        ax.text(0.02, 0.97, row_label,
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='#2E4057', alpha=0.9))


def main():
    log.info("Generating NI December loading maps...")

    # Collect data for all years
    results = {}
    for year in PLAN_YEARS:
        n, mean_l, max_l = load_and_dispatch(year, MONTH)
        if n is not None and mean_l is not None:
            ni_buses, ni_lines = get_ni_elements(n)
            results[year] = (n, mean_l, max_l, ni_buses, ni_lines)

    if not results:
        log.error("No results -- check dispatch network paths"); sys.exit(1)

    ncols = len(results)
    fig = plt.figure(figsize=(ncols * 5.5, 24))
    fig.patch.set_facecolor('white')

    for col, year in enumerate(PLAN_YEARS):
        if year not in results:
            continue
        n, mean_l, max_l, ni_buses, ni_lines = results[year]

        # Top row: mean loading
        ax_mean = make_ax(fig, 0, col, ncols)
        draw_map(ax_mean, n, mean_l, ni_buses, ni_lines, year,
                 'Mean loading')

        # Bottom row: maximum loading
        ax_max = make_ax(fig, 1, col, ncols)
        draw_map(ax_max, n, max_l, ni_buses, ni_lines, year,
                 'Maximum loading')

    # Shared colorbar
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    sm  = cm.ScalarMappable(cmap=CMAP, norm=NORM)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label('Line loading (%)', fontsize=11)
    cb.ax.tick_params(labelsize=9)
    # Mark 96.2% on colorbar
    cb.ax.axhline(96.2, color=C_N50, linewidth=2.0, linestyle='--')
    cb.ax.text(1.6, 96.2, '96.2%', fontsize=8, color=C_N50,
               fontweight='bold', va='center')

    # Legend
    legend_elements = [
        mpatches.Patch(color=C_N50,
                       label='N-50: Wellsford\u2013Marsden 110 kV (binding corridor)'),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor=C_MPE,
                   markersize=13, label='MPE: Marsden Point (proposed electrolyser)'),
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor=C_GEO,
                   markersize=9, label='KOE: Ng\u0101wh\u0101 geothermal (absorbs NI growth)'),
        mpatches.Patch(color=C_BRICK,  label='\u226590% loading'),
        mpatches.Patch(color=C_AMBER,  label='70\u201389% loading'),
        mpatches.Patch(color=C_BLUE,   label='40\u201369% loading'),
        mpatches.Patch(color=C_CREAM,  label='<40% loading'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.95,
               bbox_to_anchor=(0.45, 0.005))

    fig.suptitle(
        'North Island Network: December Line Loading  |  2025\u20132040\n'
        'Top: Mean half-hourly loading     '
        'Bottom: Maximum half-hourly loading\n'
        'N-50 peaks at 96.2% in all years (bottom row) -- '
        'Marsden Point electrolyser (MPE) lies directly beyond the saturated corridor',
        fontsize=13, fontweight='bold', y=1.002)

    fig.tight_layout(rect=[0, 0.05, 0.91, 0.995])

    out = fig_dir / 'PR6_ni_loading_maps_december.png'
    fig.savefig(str(out), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("Saved: %s", out)


if __name__ == '__main__':
    main()
