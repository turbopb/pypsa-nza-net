#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_lines_nodes.py

Cross-check lines_data.csv bus codes against nodes.csv site codes.
Reports:
  - Bus codes in lines_data not present in nodes.csv (errors)
  - JCT_ sites appearing in lines_data (should not happen)
  - Sites in nodes.csv not referenced by any line (informational)

Run from anywhere after setting PYPSA_NZA_ROOT.
"""

import os
import sys
import pandas as pd
from pathlib import Path

_root = os.environ.get("PYPSA_NZA_ROOT", "")
if not _root:
    print("ERROR: PYPSA_NZA_ROOT not set.")
    sys.exit(1)

workspace  = Path(_root) / "pypsa_nza_workspace"
manual_dir = workspace / "data" / "manual"

df_nodes = pd.read_csv(manual_dir / "nodes.csv", dtype=str)
# Use island-specific file if available, fall back to combined file
lines_file = "lines_data_si.csv"
if not (manual_dir / lines_file).exists():
    lines_file = "lines_data.csv"
df_lines = pd.read_csv(manual_dir / lines_file, dtype=str)
print(f"Lines file: {lines_file}")

for col in df_nodes.columns:
    if df_nodes[col].dtype == object:
        df_nodes[col] = df_nodes[col].str.strip()
for col in df_lines.columns:
    if df_lines[col].dtype == object:
        df_lines[col] = df_lines[col].str.strip()

all_sites = set(df_nodes["site"].str.upper())
jct_sites = {s for s in all_sites if s.startswith("JCT_")}
net_sites  = all_sites - jct_sites

# All bus endpoints in lines file
line_buses = set(df_lines["bus0"].str.upper().tolist() +
                 df_lines["bus1"].str.upper().tolist())

# Check 1: bus codes in lines not in nodes at all
missing_from_nodes = line_buses - all_sites
if missing_from_nodes:
    print("ERROR: Bus codes in lines_data.csv not found in nodes.csv:")
    for b in sorted(missing_from_nodes):
        rows = df_lines[(df_lines["bus0"].str.upper()==b) |
                        (df_lines["bus1"].str.upper()==b)]
        lines = rows["line"].tolist()
        print(f"  {b:12s}  referenced in lines: {lines}")
else:
    print("OK: All bus codes in lines_data.csv found in nodes.csv")

# Check 2: JCT_ junction nodes in lines
# JCT_ nodes are legitimate pass-through junction points used to accurately
# represent taps and tee points in the transmission line topology.
# They are not errors -- this is an informational message only.
jct_in_lines = line_buses & jct_sites
if jct_in_lines:
    print(f"\nINFO: {len(jct_in_lines)} JCT_ junction nodes used as line "
          f"endpoints (correct -- represent taps and tee points):")
    for b in sorted(jct_in_lines):
        rows = df_lines[(df_lines["bus0"].str.upper()==b) |
                        (df_lines["bus1"].str.upper()==b)]
        print(f"  {b}  in lines: {rows['line'].tolist()}")
else:
    print("OK: No JCT_ junction nodes in lines file")

# Check 3: length/distance column status
df_lines["length"]   = pd.to_numeric(df_lines["length"],   errors="coerce")
df_lines["distance_col"] = pd.to_numeric(
    df_lines.get("distance", pd.Series([None]*len(df_lines))),
    errors="coerce"
)

has_dist_col = "distance" in df_lines.columns
n_dist_nan   = df_lines["distance_col"].isna().sum() if has_dist_col else len(df_lines)
n_len_nan    = df_lines["length"].isna().sum()

print(f"\nLine length column status:")
print(f"  'length'   column: present, {n_len_nan} NaN values")
if has_dist_col:
    print(f"  'distance' column: present, {n_dist_nan} NaN / {len(df_lines)} rows")
    if n_dist_nan == 0:
        print("  -> Will use 'distance' (complete)")
    else:
        missing_dist = df_lines[df_lines["distance_col"].isna()]["line"].tolist()
        print(f"  -> 'distance' incomplete, falling back to 'length'")
        print(f"  -> Lines missing distance: {missing_dist}")
else:
    print("  'distance' column: not present -> using 'length'")

# Check 4: sites in nodes not in any line (informational, not errors)
net_not_in_lines = net_sites - line_buses - {"BGR","THW","FTB","OTB",
                                              "BEN-DC","HAY-DC"}
print(f"\nINFO: {len(net_not_in_lines)} network sites in nodes.csv "
      f"not referenced by any line:")
print(f"  (These are load-only or isolated buses -- not errors)")
for s in sorted(net_not_in_lines):
    row = df_nodes[df_nodes["site"].str.upper()==s].iloc[0]
    print(f"  {s:10s}  {row['name']:<30s}  {row['volts']:>6s} kV  {row['island']}")

print("\nValidation complete.")

# =============================================================================
# STAGE 2: NETWORK CONNECTIVITY CHECK
# =============================================================================
print("\n" + "="*60)
print("STAGE 2: NETWORK CONNECTIVITY CHECK")
print("="*60)

import networkx as nx

# Determine island scope from lines filename
ELECTRODE = {"BGR", "THW"}
DC_SITES  = {"BEN-DC", "FTB", "OTB", "HAY-DC"}

if "si" in lines_file.lower():
    check_island = "SI"
elif "ni" in lines_file.lower():
    check_island = "NI"
else:
    check_island = None

print(f"Island scope : {check_island if check_island else 'all (combined file)'}")

# Convert volts to numeric
df_lines["volts_num"] = pd.to_numeric(df_lines["volts"], errors="coerce")
df_nodes["volts_num"] = pd.to_numeric(df_nodes["volts"], errors="coerce")

# Build graph from AC lines >= 110 kV
df_lines_ac = df_lines[
    df_lines["volts_num"].notna() &
    (df_lines["volts_num"] >= 110) &
    (df_lines["volts_num"] < 350)
].copy()

# Filter nodes to island scope, >= 110 kV, excluding electrode and DC sites
# IMPORTANT: use lines file endpoints as the authoritative set of transmission
# buses, not nodes.csv voltage. nodes.csv records the substation's lowest
# voltage level which is often below 110 kV even for transmission substations.
node_mask = (
    (~df_nodes["site"].isin(ELECTRODE)) &
    (~df_nodes["site"].isin(DC_SITES)) &
    (~df_nodes["site"].str.startswith("JCT_"))
)
if check_island:
    node_mask = node_mask & (df_nodes["island"] == check_island)

df_all_nodes = df_nodes[node_mask].copy()

# Derive transmission buses from line endpoints at >= 110 kV
transmission_buses = set(df_lines_ac["bus0"].str.upper().tolist() +
                         df_lines_ac["bus1"].str.upper().tolist())
# Strip voltage suffixes (e.g. ROX_220 -> ROX) to match nodes.csv site codes
transmission_sites = set()
for b in transmission_buses:
    base = b.rsplit("_", 1)[0] if "_" in b else b
    transmission_sites.add(base)
    transmission_sites.add(b)   # keep suffixed version too

# Filter to nodes that appear in lines OR are >= 110 kV in nodes.csv
df_net_nodes = df_all_nodes[
    (df_all_nodes["site"].isin(transmission_sites)) |
    (df_all_nodes["volts_num"] >= 110)
].copy()

print(f"Network nodes: {len(df_net_nodes)}")

# Build graph -- include JCT_ nodes as pass-through nodes so that
# sites connected only via junction nodes appear correctly connected.
# JCT_ nodes are excluded from the isolated bus and degree-1 reports
# but are included in the graph edges.
G = nx.Graph()
for site in df_net_nodes["site"]:
    G.add_node(site)

for _, row in df_lines_ac.iterrows():
    b0 = str(row["bus0"]).strip().upper()
    b1 = str(row["bus1"]).strip().upper()
    G.add_edge(b0, b1, line=row["line"], volts=int(row["volts_num"]))

# Check 1: Isolated nodes -- exclude JCT_ pass-through nodes from report
isolated = sorted(
    [n for n in nx.isolates(G)
     if not n.startswith("JCT_") and
     (not check_island or
      df_nodes[df_nodes["site"]==n]["island"].values[0] == check_island
      if len(df_nodes[df_nodes["site"]==n]) > 0 else True)]
)
if isolated:
    print(f"\nWARNING: {len(isolated)} isolated {check_island or ''} buses "
          f"(no line connections):")
    for site in isolated:
        row = df_nodes[df_nodes["site"] == site]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {site:<10} {r['name']:<35} {r['volts_num']:>6.0f} kV")
else:
    print(f"\nOK: No isolated {check_island or ''} buses")

# Check 2: Connected components
# Use the full graph G (which includes JCT_ pass-through nodes) for
# connectivity detection. Filter JCT_ nodes out of the component
# reporting only -- they are real nodes in the network but are not
# substations so should not clutter the output.
components_full = list(nx.connected_components(G))

# Map each non-JCT_ node to its component
def get_reporting_components(components_full):
    """Return components filtered to non-JCT_ nodes, dropping empty ones."""
    result = []
    for comp in components_full:
        filtered = {n for n in comp if not n.startswith("JCT_")
                    and n in set(df_net_nodes["site"])}
        if filtered:
            result.append(filtered)
    return result

reporting_components = get_reporting_components(components_full)
print(f"\nConnected components ({check_island or 'all'}): "
      f"{len(reporting_components)}")
if len(reporting_components) == 1:
    print("OK: Network is fully connected (single component)")
else:
    print("WARNING: Network has multiple disconnected subgraphs:")
    for i, comp in enumerate(
            sorted(reporting_components, key=len, reverse=True)):
        print(f"  Component {i+1}: {len(comp)} buses -- "
              f"{sorted(comp)[:8]}{'...' if len(comp) > 8 else ''}")

# Build H for degree-1 check -- subgraph of non-JCT_ SI nodes
# Use the reporting components to find which nodes are truly connected
si_non_jct = {n for n in G.nodes()
              if not n.startswith("JCT_") and n in set(df_net_nodes["site"])}
H = G.subgraph(si_non_jct)

# Check 3: Degree-1 nodes -- exclude JCT_ pass-through nodes
degree_one = [n for n, d in H.degree()
              if d == 1 and not n.startswith("JCT_")]
if degree_one:
    print(f"\nINFO: {len(degree_one)} degree-1 buses (radial endpoints):")
    for site in sorted(degree_one):
        nbrs = list(H.neighbors(site))
        row  = df_nodes[df_nodes["site"] == site]
        name = row.iloc[0]["name"] if not row.empty else ""
        print(f"  {site:<10} {name:<30} -> connects to: {nbrs[0]}")
else:
    print("\nOK: No degree-1 (radial endpoint) buses")

# Check 4: Buses at >= 110 kV in nodes.csv not in any line (island-filtered)
all_line_buses = set(df_lines_ac["bus0"].str.upper().tolist() +
                     df_lines_ac["bus1"].str.upper().tolist())
in_nodes_not_lines = (
    set(df_net_nodes["site"].str.upper()) - all_line_buses - ELECTRODE - DC_SITES
)
in_nodes_not_lines = {s for s in in_nodes_not_lines
                      if not s.startswith("JCT_")}
if in_nodes_not_lines:
    print(f"\nINFO: {len(in_nodes_not_lines)} {check_island or ''} buses at "
          f">=110 kV not referenced in any line:")
    print("  (Load-only, missing lines, or correctly excluded)")
    for site in sorted(in_nodes_not_lines):
        row = df_nodes[df_nodes["site"] == site]
        if not row.empty:
            r = row.iloc[0]
            print(f"  {site:<10} {r['name']:<35} {r['volts_num']:>6.0f} kV")

print("\nConnectivity check complete.")
