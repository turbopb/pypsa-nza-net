#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
customise_combined_networks.py

Runs nza_custom_net.py customisation on the combined (unprefixed) monthly
networks only, by importing process_monthly_network directly.

Run from pypsa-nza-net directory:
    python customise_combined_networks.py
"""
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# Import from nza_custom_net
sys.path.insert(0, str(Path(__file__).parent))
import nza_custom_net as cx

_root_str = os.environ.get('PYPSA_NZA_ROOT', '')
if not _root_str:
    log.error("PYPSA_NZA_ROOT not set"); sys.exit(1)
sys.path.insert(0, str(Path(_root_str).resolve()))
from nza_root import WORKSPACE

config_path = WORKSPACE / "config" / "nza_net_config.yaml"
config = cx.load_config(config_path)
paths  = cx.resolve_paths(config)

start_year = config["scenario_years"]["start"]
end_year   = config["scenario_years"]["end"]
years      = list(range(start_year, end_year + 1))
scenario   = config.get("scenario", "reference")

net_root = paths["networks"] / scenario

for year in years:
    log.info("Year %d", year)
    for month_num in range(1, 13):
        import calendar
        month_name = calendar.month_abbr[month_num].lower()
        base_label = f"{month_name}_{year}"
        combined_dir = net_root / str(year) / base_label

        if not combined_dir.exists():
            log.warning("  Not found: %s", combined_dir)
            continue

        # Temporarily rename si_ and ni_ folders so process_monthly_network
        # falls through to the combined network
        si_dir = net_root / str(year) / f"si_{base_label}"
        ni_dir = net_root / str(year) / f"ni_{base_label}"
        si_tmp = net_root / str(year) / f"_tmp_si_{base_label}"
        ni_tmp = net_root / str(year) / f"_tmp_ni_{base_label}"

        renamed_si = False
        renamed_ni = False
        try:
            if si_dir.exists():
                si_dir.rename(si_tmp); renamed_si = True
            if ni_dir.exists():
                ni_dir.rename(ni_tmp); renamed_ni = True

            cx.process_monthly_network(month_num, year, config, paths)

        except Exception as exc:
            log.error("  Error %s/%s: %s", year, month_name, exc)
            import traceback; traceback.print_exc()
        finally:
            # Always restore
            if renamed_si and si_tmp.exists():
                si_tmp.rename(si_dir)
            if renamed_ni and ni_tmp.exists():
                ni_tmp.rename(ni_dir)

log.info("Done.")
