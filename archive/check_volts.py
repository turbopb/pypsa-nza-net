# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:27:16 2026

@author: OEM
"""

import csv
nodes = {r['site']: r for r in csv.DictReader(open(
    'C:/Users/Public/Documents/Thesis/analysis/pypsa_nza_workspace/data/manual/nodes.csv',
    encoding='utf-8-sig'))}
for site in ['WRK', 'ARA', 'NAP', 'JCT_THI_WRK', 'PAK', 'OTA', 'MNI', 'MKT', 'WIR']:
    n = nodes.get(site, None)
    if n:
        print(f"{site}: volts={n['volts']}  island={n['island']}")
    else:
        print(f"{site}: NOT FOUND")