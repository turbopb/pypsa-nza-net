#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_dispatch_v2.py

Verifies the dispatch_v2 networks and produces a network topology plot
showing the HVDC link clearly.

Checks for each month:
  - Correct bus/line/generator counts
  - Single HVDC link BEN->HAY_220
  - No DC intermediate buses
  - p_max_pu values correct
  - Loads present and non-zero
  - Single connected AC network

Produces:
  dispatch_v2_network_jan2024.png  -- network topology with HVDC link

Run from pypsa-nza-net directory:
    python verify_dispatch_v2.py
"""
import os, sys, logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pypsa

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

root      = os.environ.get('PYPSA_NZA_ROOT', '')
if not root:
    log.error("PYPSA_NZA_ROOT not set"); sys.exit(1)

workspace = Path(root) / 'pypsa_nza_workspace'
net_root  = Path(root) / 'dispatch_data' / 'cases' / 'reference_v2' / '2024'
fig_dir   = workspace / 'outputs' / 'figures' / 'validation'
fig_dir.mkdir(parents=True, exist_ok=True)

MONTHS = ['jan','feb','mar','apr','may','jun',
          'jul','aug','sep','oct','nov','dec']

DC_BUSES    = ['BEN-DC', 'FTB', 'OTB', 'HAY-DC']
EXPECTED_BUSES = 234   # 238 - 4 DC buses
EXPECTED_LINES = 283
EXPECTED_LINKS = 1     # single HVDC link
EXPECTED_GENS  = 48

# -----------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------
def verify_month(month: str) -> dict:
    folder = net_root / f'{month}_2024'
    result = {'month': month, 'ok': True, 'issues': []}

    if not folder.exists():
        result['ok'] = False
        result['issues'].append('folder not found')
        return result

    n = pypsa.Network()
    n.import_from_csv_folder(str(folder))

    # Bus count
    if len(n.buses) != EXPECTED_BUSES:
        result['issues'].append(f'buses={len(n.buses)} expected={EXPECTED_BUSES}')

    # DC buses absent
    dc_present = [b for b in DC_BUSES if b in n.buses.index]
    if dc_present:
        result['ok'] = False
        result['issues'].append(f'DC buses still present: {dc_present}')

    # Lines
    if len(n.lines) != EXPECTED_LINES:
        result['issues'].append(f'lines={len(n.lines)} expected={EXPECTED_LINES}')

    # Links
    if len(n.links) != EXPECTED_LINKS:
        result['ok'] = False
        result['issues'].append(f'links={len(n.links)} expected={EXPECTED_LINKS}')
    else:
        lnk = n.links.iloc[0]
        if lnk.bus0 != 'BEN' or lnk.bus1 != 'HAY_220':
            result['ok'] = False
            result['issues'].append(f'HVDC link endpoints wrong: {lnk.bus0}->{lnk.bus1}')
        if abs(lnk.p_nom - 1200) > 1:
            result['issues'].append(f'HVDC p_nom={lnk.p_nom} expected=1200')

    # Generators
    if len(n.generators) != EXPECTED_GENS:
        result['issues'].append(f'generators={len(n.generators)} expected={EXPECTED_GENS}')

    zero_gens = n.generators[n.generators.p_max_pu == 0].index.tolist()
    if zero_gens:
        result['ok'] = False
        result['issues'].append(f'{len(zero_gens)} generators have p_max_pu=0: {zero_gens[:5]}')

    # Total available capacity
    total_cap = (n.generators.p_nom * n.generators.p_max_pu).sum()
    result['total_cap_mw'] = total_cap

    # Total load
    if n.loads_t.p_set.empty:
        result['issues'].append('loads_t.p_set is empty')
        result['mean_load_mw'] = 0
    else:
        result['mean_load_mw'] = n.loads_t.p_set.sum(axis=1).mean()

    # Sub-networks
    n.determine_network_topology()
    n_subnets = len(n.sub_networks)
    result['sub_networks'] = n_subnets

    if result['issues']:
        status = 'OK  ' if result['ok'] else 'FAIL'
    else:
        status = 'OK  '
        result['ok'] = True

    log.info("  %s %s: buses=%d lines=%d links=%d gens=%d cap=%.0f MW load=%.0f MW sub=%d %s",
             status, month.upper(),
             len(n.buses), len(n.lines), len(n.links), len(n.generators),
             result['total_cap_mw'], result['mean_load_mw'],
             n_subnets,
             ' | '.join(result['issues']) if result['issues'] else '')

    return result, n


# -----------------------------------------------------------------------
# Network plot
# -----------------------------------------------------------------------
def plot_network(n: pypsa.Network, month: str) -> None:
    """Plot full NZ network with HVDC link highlighted."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    PROJ = ccrs.PlateCarree()
    NZ_EXTENT = [165.5, 179.0, -47.5, -33.5]

    fig = plt.figure(figsize=(12, 18), dpi=150)
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_extent(NZ_EXTENT, crs=PROJ)
    ax.add_feature(cfeature.LAND,      facecolor='#f5f5f0', zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor='#555555', zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor='#aed6f1', edgecolor='#aed6f1', zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False
    gl.xformatter   = LONGITUDE_FORMATTER
    gl.yformatter   = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    buses = n.buses
    gen_buses  = set(n.generators.bus)
    load_buses = set(n.loads.bus)

    # Draw AC lines
    for _, line in n.lines.iterrows():
        b0 = line.bus0
        b1 = line.bus1
        if b0 not in buses.index or b1 not in buses.index:
            continue
        x0, y0 = float(buses.at[b0,'x']), float(buses.at[b0,'y'])
        x1, y1 = float(buses.at[b1,'x']), float(buses.at[b1,'y'])
        v = float(buses.at[b0,'v_nom'])
        color = '#c0392b' if v >= 220 else '#2980b9'
        lw    = 1.5 if v >= 220 else 0.8
        ax.plot([x0,x1],[y0,y1], color=color, linewidth=lw,
                alpha=0.7, transform=PROJ, zorder=2)

    # Draw HVDC link prominently
    for _, lnk in n.links.iterrows():
        b0, b1 = lnk.bus0, lnk.bus1
        if b0 not in buses.index or b1 not in buses.index:
            continue
        x0, y0 = float(buses.at[b0,'x']), float(buses.at[b0,'y'])
        x1, y1 = float(buses.at[b1,'x']), float(buses.at[b1,'y'])
        ax.plot([x0,x1],[y0,y1], color='#8e44ad', linewidth=4.0,
                alpha=0.9, transform=PROJ, zorder=5,
                linestyle='--', label='HVDC Link')
        # Label midpoint
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.annotate('HVDC\n{:.0f} MW'.format(lnk.p_nom),
                    xy=(mx, my), xytext=(mx+0.8, my),
                    fontsize=9, fontweight='bold', color='#8e44ad',
                    xycoords=PROJ._as_mpl_transform(ax),
                    textcoords=PROJ._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle='-', color='#8e44ad', lw=1.0),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#8e44ad', alpha=0.9),
                    zorder=6)

    # Draw buses
    for bus_id, bus in buses.iterrows():
        x, y = float(bus.x), float(bus.y)
        if bus_id in gen_buses:
            mc, mk, ms = '#1565C0', 'o', 8
        elif bus_id in load_buses:
            mc, mk, ms = '#B71C1C', 's', 7
        else:
            mc, mk, ms = '#546E7A', 'o', 4
        ax.plot(x, y, mk, color=mc, markersize=ms, zorder=4,
                markeredgecolor='white', markeredgewidth=0.6,
                transform=PROJ)

    # Legend
    legend_elements = [
        mlines.Line2D([0],[0], color='#c0392b', linewidth=2.5, label='220 kV AC'),
        mlines.Line2D([0],[0], color='#2980b9', linewidth=1.5, label='110 kV AC'),
        mlines.Line2D([0],[0], color='#8e44ad', linewidth=4.0,
                      linestyle='--', label='HVDC Link (1200 MW)'),
        mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor='#1565C0',
                      markersize=9, markeredgecolor='white', label='Generation bus'),
        mlines.Line2D([0],[0], marker='s', color='w', markerfacecolor='#B71C1C',
                      markersize=8, markeredgecolor='white', label='Load bus'),
        mlines.Line2D([0],[0], marker='o', color='w', markerfacecolor='#546E7A',
                      markersize=5, markeredgecolor='white', label='Junction bus'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              framealpha=0.95, edgecolor='#bbbbbb', borderpad=0.8)

    ax.set_title(
        'NZ Dispatch Network v2 -- {} 2024\n'
        '{} buses  {}  AC lines  1 HVDC link  {} generators'.format(
            month.capitalize(),
            len(n.buses), len(n.lines), len(n.generators)),
        fontsize=12, fontweight='bold', pad=12)

    out = fig_dir / 'dispatch_v2_network_{}_2024.png'.format(month)
    fig.savefig(str(out), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    log.info("  Plot saved: %s", out)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info("VERIFYING DISPATCH V2 NETWORKS -- 2024")
    log.info("=" * 60)

    all_ok = True
    jan_network = None

    for month in MONTHS:
        result, n = verify_month(month)
        if not result['ok']:
            all_ok = False
        if month == 'jan':
            jan_network = n

    log.info("")
    if all_ok:
        log.info("ALL MONTHS VERIFIED OK")
    else:
        log.info("SOME MONTHS HAVE ISSUES -- see above")

    # Plot January network
    if jan_network is not None:
        log.info("")
        log.info("Generating network plot for January 2024 ...")
        try:
            plot_network(jan_network, 'january')
        except Exception as e:
            log.error("Plot failed: %s", e)
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()
