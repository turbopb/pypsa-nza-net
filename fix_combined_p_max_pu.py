#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fix_combined_p_max_pu.py

Sets correct p_max_pu values in NI and combined network generators.csv
for all 24 months, using carrier-based availability from nza_net_config.yaml.

nza_custom_net.py only processes SI networks. This script applies the same
availability scalars to NI and combined networks directly.

Run from pypsa-nza-net directory:
    python fix_combined_p_max_pu.py
"""
import pandas as pd
import yaml
import os
from pathlib import Path

# Availability scalars from nza_net_config.yaml
# Wind and solar p_max_pu are set to 1.0 here (dispatch solver uses
# generator timeseries from EA data via p_set, not p_max_pu for renewables)
AVAILABILITY = {
    'hydro':      0.75,
    'geothermal': 0.87,
    'gas':        1.00,
    'ocgt':       1.00,
    'ccgt':       1.00,
    'coal_gas':   1.00,
    'diesel':     1.00,
    'wind':       1.00,
    'solar_pv':   1.00,
}

root = os.environ.get('PYPSA_NZA_ROOT', '')
if not root:
    print("ERROR: PYPSA_NZA_ROOT not set"); exit(1)

workspace = Path(root) / 'pypsa_nza_workspace'
net_root  = workspace / 'models' / 'networks' / 'reference'

MONTHS = ['jan','feb','mar','apr','may','jun',
          'jul','aug','sep','oct','nov','dec']

# Networks to fix: NI and combined (SI already done by nza_custom_net.py)
PREFIXES = ['ni_', '']   # ni_jan_2024 and jan_2024

fixed = 0
errors = 0

for year in [2024, 2025]:
    for month in MONTHS:
        for prefix in PREFIXES:
            folder = net_root / str(year) / f'{prefix}{month}_{year}'
            if not folder.exists():
                continue
            gen_file = folder / 'generators.csv'
            if not gen_file.exists():
                continue
            try:
                df = pd.read_csv(str(gen_file))
                if 'carrier' not in df.columns:
                    print(f'SKIP {prefix}{year}/{month}: no carrier column')
                    continue

                # Apply availability by carrier
                df['p_max_pu'] = df['carrier'].map(AVAILABILITY).fillna(0.75)

                df.to_csv(str(gen_file), index=False)
                max_val = df['p_max_pu'].max()
                print(f'OK   {prefix}{year}/{month}: {len(df)} generators, max p_max_pu={max_val:.2f}')
                fixed += 1
            except Exception as e:
                print(f'ERROR {prefix}{year}/{month}: {e}')
                errors += 1

print(f'\nDone. Fixed: {fixed}  Errors: {errors}')
