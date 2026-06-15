#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_dispatch_v2.py

Builds clean dispatch networks for 2024 from the validated combined
SI+NI network in pypsa_nza_workspace.

Changes from base network:
  - Removes three-segment HVDC chain (BEN-DC, FTB, OTB, HAY-DC)
  - Adds single HVDC Link: BEN -> HAY_220, p_nom=1200 MW, eta=0.965
  - Sets p_max_pu by carrier:
      hydro:      0.75
      geothermal: 0.87
      all others: 1.00
  - Saves to dispatch_data/cases/reference_v2/2024/{month}_2024/

Run from pypsa-nza-net directory:
    python build_dispatch_v2.py

Output:
    dispatch_data/cases/reference_v2/2024/{month}_2024/
"""
import os, sys, logging
from pathlib import Path
import pandas as pd
import pypsa

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# Paths
root      = os.environ.get('PYPSA_NZA_ROOT', '')
if not root:
    log.error("PYPSA_NZA_ROOT not set"); sys.exit(1)

workspace = Path(root) / 'pypsa_nza_workspace'
src_root  = workspace / 'models' / 'networks' / 'reference' / '2024'
dst_root  = Path(root) / 'dispatch_data' / 'cases' / 'reference_v2' / '2024'

MONTHS = ['jan','feb','mar','apr','may','jun',
          'jul','aug','sep','oct','nov','dec']

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

def build_month(month: str) -> None:
    src_dir = src_root / f'{month}_2024'
    dst_dir = dst_root / f'{month}_2024'

    if not src_dir.exists():
        log.warning("Source not found: %s", src_dir)
        return

    log.info("Building %s_2024 ...", month)

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

    # --- Add AC proxy line BEN -> HAY_220 (diagnostic test) ---
    # Using AC line instead of HVDC Link to test whether the LP solver
    # can route power across the Cook Strait connection.
    # If this works, the problem is with the HVDC Link formulation.
    # If this fails, the problem is with the island networks themselves.
    if HVDC_BUS0 in n.buses.index and HVDC_BUS1 in n.buses.index:
        n.add('Line', 'HVDC_AC_proxy',
              bus0=HVDC_BUS0,
              bus1=HVDC_BUS1,
              x=0.001,
              r=0.0,
              s_nom=HVDC_P_NOM,
              length=610.0)
        log.info("  Added AC proxy line: %s -> %s  s_nom=%s MW",
                 HVDC_BUS0, HVDC_BUS1, HVDC_P_NOM)
    else:
        log.warning("  BEN or HAY_220 not found -- AC proxy not added")

    # --- Set p_max_pu by carrier ---
    for gen in n.generators.index:
        carrier = n.generators.at[gen, 'carrier']
        p_max   = AVAILABILITY.get(carrier, DEFAULT_P_MAX_PU)
        n.generators.at[gen, 'p_max_pu'] = p_max

    log.info("  Generators: %d  total available: %.0f MW",
             len(n.generators),
             (n.generators.p_nom * n.generators.p_max_pu).sum())

    # --- Save ---
    dst_dir.mkdir(parents=True, exist_ok=True)
    n.export_to_csv_folder(str(dst_dir))
    log.info("  Saved: %s", dst_dir)


def main():
    log.info("Building dispatch networks for 2024 -> %s", dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    for month in MONTHS:
        try:
            build_month(month)
        except Exception as e:
            log.error("Error on %s: %s", month, e)
            import traceback; traceback.print_exc()
    log.info("Done.")

if __name__ == '__main__':
    main()
