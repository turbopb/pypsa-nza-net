#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nza_calculate_demand_scaling.py

Calculates MBIE EDGS 2024 demand scaling factors for Reference and
Innovation scenarios, relative to 2025 base year, and writes them
directly into nza_net_config.yaml and dispatch_config.yaml.

CRITICAL: All factors are relative to 2025 (not 2024).
  factor[year] = MBIE_demand[year] / MBIE_demand[2025]

The 2025 base year was selected because:
  - 2024 had two abnormal events (pylon collapse, cold snap)
  - 2025 represents normal operating conditions
  - Using 2025=1.0 ensures consistency between validation and planning

Usage:
    python nza_calculate_demand_scaling.py --dry-run   # print only
    python nza_calculate_demand_scaling.py             # update configs

Author: Phillippe Bruneau
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import yaml
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MBIE_CSV     = Path(r"C:\Users\Public\Documents\Thesis\analysis\data\demand\future"
                    r"\electricity-demand-scenarios-total-2024-results.csv")
NET_CONFIG   = Path(r"C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace"
                    r"\config\nza_net_config.yaml")
DISP_CONFIG  = Path(r"C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-dispatch"
                    r"\config\dispatch_config.yaml")

BASE_YEAR    = 2025
PLAN_YEARS   = [2025, 2030, 2035, 2040, 2045, 2050]
SCENARIOS    = ['Reference', 'Innovation']


# ---------------------------------------------------------------------------
# Calculate factors
# ---------------------------------------------------------------------------
def calculate_factors(csv_path):
    """Return {scenario: {year: factor}} relative to BASE_YEAR."""
    df  = pd.read_csv(str(csv_path))
    out = {}
    for scenario in SCENARIOS:
        s = df[df['Scenario'] == scenario].sort_values('Year')
        f = interp1d(s['Year'].values, s['Value'].values, kind='linear')
        base = float(f(BASE_YEAR))
        out[scenario] = {yr: round(float(f(yr)) / base, 6) for yr in PLAN_YEARS}
        log.info("%-12s base %d = %.4f TWh", scenario, BASE_YEAR, base)
    return out


def print_factors(factors):
    """Print factors and comparison table."""
    log.info("")
    log.info("=" * 65)
    log.info("MBIE EDGS 2024 demand scaling factors (base year = %d)", BASE_YEAR)
    log.info("=" * 65)
    log.info("  %-6s  %-12s  %-12s  %s",
             "Year", "Reference", "Innovation", "Innovation/Reference")
    log.info("  " + "-" * 55)
    for yr in PLAN_YEARS:
        r = factors['Reference'][yr]
        i = factors['Innovation'][yr]
        log.info("  %-6d  %-12.6f  %-12.6f  %.3f",
                 yr, r, i, i/r)
    log.info("=" * 65)


# ---------------------------------------------------------------------------
# Update configs
# ---------------------------------------------------------------------------
def update_net_config(factors, dry_run):
    """Add/replace demand_scaling_reference and demand_scaling_innovation
    in nza_net_config.yaml."""
    with open(NET_CONFIG) as f:
        content = f.read()

    for scenario in SCENARIOS:
        key   = f"demand_scaling_{scenario.lower()}"
        block = f"\n{key}:\n"
        for yr, val in factors[scenario].items():
            block += f"  {yr}: {val}\n"

        # Replace existing block or append
        pattern = rf"^{key}:.*?(?=^\S|\Z)"
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            content = re.sub(pattern, block.lstrip(), content,
                             flags=re.MULTILINE | re.DOTALL)
            log.info("  nza_net_config.yaml: replaced %s", key)
        else:
            content += block
            log.info("  nza_net_config.yaml: added %s", key)

    if not dry_run:
        with open(NET_CONFIG, 'w') as f:
            f.write(content)
        log.info("  Written: %s", NET_CONFIG)
    else:
        log.info("  [dry-run] would write: %s", NET_CONFIG)


def update_dispatch_config(factors, dry_run):
    """Add/replace demand_scenarios sections in dispatch_config.yaml.
    
    Writes to config['demand_scenarios']['reference'] and
    config['demand_scenarios']['innovation'] -- the keys that
    01_setup_year.py reads via config['demand_scenarios'][scenario].
    Also records the base year for documentation.
    """
    with open(DISP_CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault('demand_scenarios', {})
    cfg['demand_scenarios']['reference']  = factors['Reference']
    cfg['demand_scenarios']['innovation'] = factors['Innovation']
    cfg['demand_scenarios']['base_year']  = BASE_YEAR

    if not dry_run:
        with open(DISP_CONFIG, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        log.info("  Written: %s", DISP_CONFIG)
    else:
        log.info("  [dry-run] would write: %s", DISP_CONFIG)


# ---------------------------------------------------------------------------
# Wind year map extension
# ---------------------------------------------------------------------------
def print_wind_year_map():
    """Print the wind_data_year_map extension needed for 2035-2050."""
    log.info("")
    log.info("Wind year map extension needed in nza_net_config.yaml:")
    log.info("(cycles through available NIWA years 2004-2007)")
    available = [2004, 2005, 2006, 2007]
    log.info("")
    log.info("  wind_data_year_map:")
    cycle = available * 10
    idx   = 0
    for yr in range(2031, 2051):
        log.info("    %d: %d", yr, cycle[idx % len(available)])
        idx += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calculate and write MBIE demand scaling factors.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print factors only, do not write configs")
    args = parser.parse_args()

    log.info("MBIE EDGS 2024 demand scaling factor calculator")
    log.info("Base year: %d  |  Scenarios: %s", BASE_YEAR, SCENARIOS)
    log.info("Planning years: %s", PLAN_YEARS)
    log.info("")

    factors = calculate_factors(MBIE_CSV)
    print_factors(factors)
    print_wind_year_map()

    log.info("")
    if args.dry_run:
        log.info("Dry run -- no files written.")
    else:
        log.info("Writing to config files...")
        update_net_config(factors, dry_run=False)
        update_dispatch_config(factors, dry_run=False)
        log.info("Done.")
        log.info("")
        log.info("Next steps:")
        log.info("  1. Add wind_data_year_map entries (shown above) to nza_net_config.yaml")
        log.info("  2. Add 2045/2050 to PLANNING_SEQUENCE in run_planning_pipeline.py")
        log.info("  3. Rebuild networks: nza_base_net.py + nza_custom_net.py")
        log.info("  4. Rebuild dispatch: build_dispatch_v2.py --year 2025")
        log.info("  5. Run reference pipeline: python run_planning_pipeline.py")
        log.info("  6. Run innovation pipeline: python run_planning_pipeline.py --scenario innovation")


if __name__ == "__main__":
    main()
