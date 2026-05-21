# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:33:32 2026

@author: OEM
"""

import csv
nodes = {r['site']: r for r in csv.DictReader(open('C:/Users/Public/Documents/Thesis/analysis/pypsa_nza_workspace/data/manual/nodes.csv', encoding='utf-8-sig'))}
lines = list(csv.DictReader(open('C:/Users/Public/Documents/Thesis/analysis/pypsa_nza_workspace/data/manual/lines_data_ni.csv', encoding='utf-8-sig')))
mismatches = set()
for r in lines:
    for bus in [r['bus0'], r['bus1']]:
        if bus in nodes:
            node_v = int(nodes[bus]['volts']) if nodes[bus]['volts'].strip() else 0
            line_v = int(r['volts'])
            if node_v != line_v and not bus.startswith('JCT_'):
                mismatches.add((bus, node_v, line_v))
for m in sorted(mismatches):
    print(f'Bus {m[0]}: nodes.csv volts={m[1]}  line volts={m[2]}')
    
    
    
    