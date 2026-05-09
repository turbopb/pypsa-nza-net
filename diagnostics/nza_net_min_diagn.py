
#    nza_net_min_diagn.py

import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network("si_jan_2024")

print("PyPSA version:")
import pypsa
print(pypsa.__version__)

print("\nBuses:", len(n.buses))
print("Lines:", len(n.lines))
print("Generators:", len(n.generators))
print("Loads:", len(n.loads))
print("Snapshots:", len(n.snapshots))

# Basic consistency check
n.consistency_check()

# Check line reactances
bad_x = n.lines[(n.lines.x.isna()) | (n.lines.x == 0)]
print("\nLines with missing or zero x:")
print(bad_x[["bus0", "bus1", "x", "r", "s_nom"]])

# Check disconnected buses
all_line_buses = set(n.lines.bus0).union(set(n.lines.bus1))
isolated_buses = set(n.buses.index) - all_line_buses
print("\nBuses not connected to any line:")
print(isolated_buses)

# Check generator and load buses exist
missing_gen_buses = set(n.generators.bus) - set(n.buses.index)
missing_load_buses = set(n.loads.bus) - set(n.buses.index)

print("\nGenerator buses missing from n.buses:")
print(missing_gen_buses)

print("\nLoad buses missing from n.buses:")
print(missing_load_buses)