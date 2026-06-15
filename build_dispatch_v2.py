#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_dispatch_v2.py

Builds clean dispatch networks from the validated combined SI+NI network.

Changes from base network:
  - Removes DC buses if present (BEN-DC, FTB, OTB, HAY-DC)
  - Network now built with single HVDC Link BEN->HAY_220 by nza_base_net.py
  - Reads s_nom from line_loading.csv (PyPSA does not persist s_nom to CSV)
  - Fixes loads-p_set.csv datetime index
  - Reads p_max_pu from nza_net_config.yaml (seasonal hydro profile)

Usage:
    python build_dispatch_v2.py [--year 2024] [--year 2025]
    python build_dispatch_v2.py --year 2024
    python build_dispatch_v2.py --year 2025

Output:
    dispatch_data/cases/reference_v2/{year}/{month}_{year}/
"""
import os, sys, logging, argparse, yaml
from pathlib import Path
import pandas as pd
import pypsa

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# Paths
root = os.environ.get('PYPSA_NZA_ROOT', '')
if not root:
    log.error("PYPSA_NZA_ROOT not set"); sys.exit(1)

workspace = Path(root) / 'pypsa_nza_workspace'

MONTHS = ['jan','feb','mar','apr','may','jun',
          'jul','aug','sep','oct','nov','dec']

# Month number lookup
MONTH_NUM = {m: i+1 for i, m in enumerate(MONTHS)}

# DC buses to remove
DC_BUSES = ['BEN-DC', 'FTB', 'OTB', 'HAY-DC']

# HVDC proxy link parameters
HVDC_P_NOM    = 1200.0   # MW
HVDC_ETA      = 0.965    # combined converter + cable efficiency
HVDC_BUS0     = 'BEN'
HVDC_BUS1     = 'HAY_220'

# Generator availability by carrier
AVAILABILITY = {
    'hydro':      0.75,
    'geothermal': 0.87,
}
DEFAULT_P_MAX_PU = 1.00   # all other carriers

def build_month(month: str, year: int, src_root: Path, dst_root: Path,
                hydro_seasonal: dict) -> None:
    src_dir = src_root / f'{month}_{year}'
    dst_dir = dst_root / f'{month}_{year}'

    if not src_dir.exists():
        log.warning("Source not found: %s", src_dir)
        return

    log.info("Building %s_%d ...", month, year)

    # Load combined network
    n = pypsa.Network()
    n.import_from_csv_folder(str(src_dir))

    # --- Remove HVDC Links ---
    if not n.links.empty:
        dc_links = n.links[
            n.links.bus0.isin(DC_BUSES) | n.links.bus1.isin(DC_BUSES)
        ].index.tolist()
        if dc_links:
            n.remove('Link', dc_links)
            log.info("  Removed HVDC links: %s", dc_links)

    # --- Remove DC buses ---
    existing_dc = [b for b in DC_BUSES if b in n.buses.index]
    if existing_dc:
        n.remove('Bus', existing_dc)
        log.info("  Removed DC buses: %s", existing_dc)

    # --- Add single HVDC Link BEN -> HAY_220 ---
    if HVDC_BUS0 in n.buses.index and HVDC_BUS1 in n.buses.index:
        n.add('Link', 'HVDC_BEN_HAY',
              bus0=HVDC_BUS0,
              bus1=HVDC_BUS1,
              p_nom=HVDC_P_NOM,
              p_min_pu=-1.0,
              p_max_pu=1.0,
              efficiency=HVDC_ETA,
              )
        log.info("  Added HVDC Link: %s -> %s  p_nom=%s MW  eta=%.3f",
                 HVDC_BUS0, HVDC_BUS1, HVDC_P_NOM, HVDC_ETA)
    else:
        log.warning("  BEN or HAY_220 not found -- HVDC link not added")

    # --- Set p_max_pu by carrier (seasonal hydro from config) ---
    month_num = MONTH_NUM[month]
    for gen in n.generators.index:
        carrier = n.generators.at[gen, 'carrier']
        if carrier == 'hydro' and hydro_seasonal:
            p_max = float(hydro_seasonal.get(month_num,
                          AVAILABILITY.get('hydro', 0.75)))
        else:
            p_max = AVAILABILITY.get(carrier, DEFAULT_P_MAX_PU)
        n.generators.at[gen, 'p_max_pu'] = p_max
    hydro_pu = hydro_seasonal.get(month_num, AVAILABILITY.get('hydro', 0.75))
    log.info("  Hydro p_max_pu (month %d): %.2f", month_num, hydro_pu)

    # --- Fix s_nom from validation results ---
    # PyPSA stores line ratings via type strings and calculates s_nom at
    # runtime, but does not persist it to CSV. Read from line_loading.csv
    # produced by the DC power flow validation which has s_nom_MW per line.
    loading_path = (workspace / "outputs" / "verification_pf" / "combined"
                    / str(year) / f"{month}_{year}" / "line_loading.csv")
    if loading_path.exists():
        loading = pd.read_csv(str(loading_path), index_col=0)
        if "s_nom_MW" in loading.columns:
            common = n.lines.index.intersection(loading.index)
            n.lines.loc[common, "s_nom"] = loading.loc[common, "s_nom_MW"]
            zero_left = (n.lines.s_nom == 0).sum()
            log.info("  s_nom set from validation: min=%.0f  max=%.0f  mean=%.0f MW  zeros=%d",
                     n.lines.s_nom.min(), n.lines.s_nom.max(),
                     n.lines.s_nom.mean(), zero_left)
        else:
            log.warning("  s_nom_MW column not in line_loading.csv")
    else:
        log.warning("  line_loading.csv not found: %s", loading_path)

    log.info("  Generators: %d  total available: %.0f MW",
             len(n.generators),
             (n.generators.p_nom * n.generators.p_max_pu).sum())

    # --- Save ---
    dst_dir.mkdir(parents=True, exist_ok=True)
    n.export_to_csv_folder(str(dst_dir))

    # Fix loads-p_set.csv: PyPSA 1.0.3 writes integer index,
    # PyPSA 1.2.2 requires datetime index matching network snapshots.
    p_set_path = dst_dir / "loads-p_set.csv"
    if p_set_path.exists():
        df = pd.read_csv(str(p_set_path), index_col=0)
        df.index = n.snapshots
        df.to_csv(str(p_set_path))
        log.info("  Fixed loads-p_set.csv datetime index")

    log.info("  Saved: %s", dst_dir)


def main():
    parser = argparse.ArgumentParser(description="Build dispatch networks")
    parser.add_argument('--year', type=int, default=2025,
                        help='Source year to build (default: 2025)')
    args = parser.parse_args()
    year = args.year

    src_root = workspace / 'models' / 'networks' / 'reference' / str(year)
    dst_root = Path(root) / 'dispatch_data' / 'cases' / 'reference_v2' / str(year)

    # Load seasonal hydro profile from config
    config_path = workspace / 'config' / 'nza_net_config.yaml'
    hydro_seasonal = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        raw = cfg.get('hydro_seasonal_availability', {})
        hydro_seasonal = {int(k): float(v) for k, v in raw.items()}
        log.info("Loaded seasonal hydro profile: %s", hydro_seasonal)
    else:
        log.warning("Config not found: %s -- using static availability", config_path)

    log.info("Building dispatch networks for %d -> %s", year, dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        try:
            build_month(month, year, src_root, dst_root, hydro_seasonal)
        except Exception as e:
            log.error("Error on %s: %s", month, e)
            import traceback; traceback.print_exc()
    log.info("Done.")

if __name__ == '__main__':
    main()
