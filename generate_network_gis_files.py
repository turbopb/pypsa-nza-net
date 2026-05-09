#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_network_gis_files.py

Produce GIS-ready CSV files from the pypsa-nza-net static data.
Run this from anywhere after setting PYPSA_NZA_ROOT.

Outputs (written to pypsa_nza_workspace/data/processed/static/):
    network_buses.csv       -- all PyPSA buses with coordinates
    transformers.csv        -- transformer locations
    network_lines.csv       -- transmission lines with endpoint coordinates
    network_links.csv       -- HVDC link segments with endpoint coordinates
"""

import os
import sys
import pandas as pd
from pathlib import Path

# -- Workspace setup -------------------------------------------------------
_root = os.environ.get("PYPSA_NZA_ROOT", "")
if not _root:
    print("ERROR: PYPSA_NZA_ROOT not set. See SETUP.md.")
    sys.exit(1)

workspace  = Path(_root) / "pypsa_nza_workspace"
manual_dir = workspace / "data" / "manual"
out_dir    = workspace / "data" / "processed" / "static"
out_dir.mkdir(parents=True, exist_ok=True)

# -- Load static inputs ----------------------------------------------------
df_nodes = pd.read_csv(manual_dir / "nodes.csv", dtype=str)

# Use island-specific lines files if available, fall back to combined file
lines_files = []
for fname in ("lines_data_si.csv", "lines_data_ni.csv", "lines_data.csv"):
    fpath = manual_dir / fname
    if fpath.exists() and "old" not in fname:
        lines_files.append(fpath)
        if fname == "lines_data.csv":
            break  # combined file covers everything

if not lines_files:
    print("ERROR: No lines data file found in", manual_dir)
    sys.exit(1)

df_lines = pd.concat(
    [pd.read_csv(f, dtype=str) for f in lines_files],
    ignore_index=True
)
print(f"Lines loaded from: {[f.name for f in lines_files]} ({len(df_lines)} rows)")

df_links = pd.read_csv(manual_dir / "links_data.csv", dtype=str)

for col in df_nodes.columns:
    df_nodes[col] = df_nodes[col].str.strip()
df_nodes["volts"] = pd.to_numeric(df_nodes["volts"], errors="coerce")
df_nodes["long"]  = pd.to_numeric(df_nodes["long"],  errors="coerce")
df_nodes["lat"]   = pd.to_numeric(df_nodes["lat"],   errors="coerce")
df_nodes["X"]     = pd.to_numeric(df_nodes["X"],     errors="coerce")
df_nodes["Y"]     = pd.to_numeric(df_nodes["Y"],     errors="coerce")

df_lines["volts"]    = pd.to_numeric(df_lines["volts"],    errors="coerce")
df_lines["distance"] = pd.to_numeric(df_lines["distance"], errors="coerce")
df_links["length"]   = pd.to_numeric(df_links["length"],   errors="coerce")

ELECTRODE_SITES = {"BGR", "THW"}
df_nodes = df_nodes[~df_nodes["site"].isin(ELECTRODE_SITES)].copy()

node_lkp = df_nodes.set_index("site")[["name","island","zone","long","lat","X","Y"]]

# -- Detect multi-voltage sites from lines ---------------------------------
AC_LINES = df_lines[df_lines["volts"].notna() & (df_lines["volts"] < 350)].copy()
site_volts = {}
for _, row in AC_LINES.iterrows():
    v = int(row["volts"])
    for col in ("bus0", "bus1"):
        s = str(row[col]).strip().upper()
        site_volts.setdefault(s, set()).add(v)

multi_volt = {s for s, vs in site_volts.items() if len(vs) > 1}
dc_sites   = set(df_links["bus0"].str.strip().tolist() +
                  df_links["bus1"].str.strip().tolist())

def bus_name(site, volts):
    if site in multi_volt:
        return f"{site}_{int(volts)}"
    return site

# ==========================================================================
# FILE 1: network_buses.csv
# ==========================================================================
bus_rows = []
for _, row in df_nodes.iterrows():
    site  = row["site"]
    volts = row["volts"]
    
    if site in dc_sites:
        bus_rows.append(dict(bus_name=site, site=site, name=row["name"],
                             zone=row["zone"], island=row["island"],
                             v_nom_kv=350, carrier="DC",
                             bus_type="DC_converter",
                             long=row["long"], lat=row["lat"],
                             nztm_e=row["X"], nztm_n=row["Y"]))
        continue

    if site in multi_volt:
        for v in sorted(site_volts.get(site, [220, 110]), reverse=True):
            bus_rows.append(dict(bus_name=f"{site}_{v}", site=site,
                                 name=row["name"], zone=row["zone"],
                                 island=row["island"], v_nom_kv=v,
                                 carrier="AC", bus_type="multi_voltage",
                                 long=row["long"], lat=row["lat"],
                                 nztm_e=row["X"], nztm_n=row["Y"]))
        continue

    if pd.notna(volts):
        bus_rows.append(dict(bus_name=site, site=site, name=row["name"],
                             zone=row["zone"], island=row["island"],
                             v_nom_kv=int(volts), carrier="AC",
                             bus_type="single_voltage",
                             long=row["long"], lat=row["lat"],
                             nztm_e=row["X"], nztm_n=row["Y"]))

df_buses_out = pd.DataFrame(bus_rows)
df_buses_out.to_csv(out_dir / "network_buses.csv", index=False)
print(f"network_buses.csv:  {len(df_buses_out)} buses")

# ==========================================================================
# FILE 2: transformers.csv
# ==========================================================================
trafo_rows = []
for site in sorted(multi_volt):
    vs = sorted(site_volts.get(site, [220, 110]), reverse=True)
    v_high, v_low = vs[0], vs[-1]
    if site not in node_lkp.index:
        print(f"  WARNING: {site} not in nodes.csv")
        continue
    n = node_lkp.loc[site]
    trafo_rows.append(dict(
        trafo_name=f"TRAFO_{site}_{v_high}_{v_low}",
        site=site, name=n["name"],
        zone=n["zone"], island=n["island"],
        bus_high=f"{site}_{v_high}",
        bus_low=f"{site}_{v_low}",
        v_high_kv=v_high, v_low_kv=v_low,
        trafo_type="630MVA 220/110kV",
        s_nom_extendable=True,
        capital_cost_placeholder=0.0,
        long=n["long"], lat=n["lat"],
        nztm_e=n["X"], nztm_n=n["Y"],
    ))

df_trafo_out = pd.DataFrame(trafo_rows)
df_trafo_out.to_csv(out_dir / "transformers.csv", index=False)
print(f"transformers.csv:   {len(df_trafo_out)} transformers")

# ==========================================================================
# FILE 3: network_lines.csv
# ==========================================================================
line_rows = []
for _, row in AC_LINES[AC_LINES["volts"] >= 110].iterrows():
    b0  = str(row["bus0"]).strip().upper()
    b1  = str(row["bus1"]).strip().upper()
    v   = int(row["volts"])
    cfg = str(row.get("config", row.get("cofing", ""))).strip().lower()

    b0_name = bus_name(b0, v)
    b1_name = bus_name(b1, v)

    # Endpoint coordinates
    long0 = node_lkp.loc[b0, "long"] if b0 in node_lkp.index else None
    lat0  = node_lkp.loc[b0, "lat"]  if b0 in node_lkp.index else None
    long1 = node_lkp.loc[b1, "long"] if b1 in node_lkp.index else None
    lat1  = node_lkp.loc[b1, "lat"]  if b1 in node_lkp.index else None
    e0    = node_lkp.loc[b0, "X"]    if b0 in node_lkp.index else None
    n0    = node_lkp.loc[b0, "Y"]    if b0 in node_lkp.index else None
    e1    = node_lkp.loc[b1, "X"]    if b1 in node_lkp.index else None
    n1    = node_lkp.loc[b1, "Y"]    if b1 in node_lkp.index else None

    line_rows.append(dict(
        line_name=row["line"],
        description=row.get("description",""),
        bus0_site=b0, bus1_site=b1,
        bus0=b0_name, bus1=b1_name,
        ss0=row.get("ss0",""), ss1=row.get("ss1",""),
        volts_kv=v,
        config=cfg,
        num_parallel=2 if cfg == "double" else 1,
        distance_km=row["distance"],
        length_straight_km=row.get("length",""),
        long0=long0, lat0=lat0, long1=long1, lat1=lat1,
        nztm_e0=e0, nztm_n0=n0, nztm_e1=e1, nztm_n1=n1,
        length_note="distance_km used for impedance; straight-line lower bound",
    ))

df_lines_out = pd.DataFrame(line_rows)
df_lines_out.to_csv(out_dir / "network_lines.csv", index=False)
print(f"network_lines.csv:  {len(df_lines_out)} lines")
print(f"  220 kV: {(df_lines_out['volts_kv']==220).sum()}")
print(f"  110 kV: {(df_lines_out['volts_kv']==110).sum()}")
print(f"  double-circuit: {(df_lines_out['num_parallel']==2).sum()}")

# ==========================================================================
# FILE 4: network_links.csv
# ==========================================================================
link_rows = []
for _, row in df_links.iterrows():
    b0 = str(row["bus0"]).strip()
    b1 = str(row["bus1"]).strip()
    long0 = node_lkp.loc[b0, "long"] if b0 in node_lkp.index else None
    lat0  = node_lkp.loc[b0, "lat"]  if b0 in node_lkp.index else None
    long1 = node_lkp.loc[b1, "long"] if b1 in node_lkp.index else None
    lat1  = node_lkp.loc[b1, "lat"]  if b1 in node_lkp.index else None
    e0    = node_lkp.loc[b0, "X"]    if b0 in node_lkp.index else None
    n0    = node_lkp.loc[b0, "Y"]    if b0 in node_lkp.index else None
    e1    = node_lkp.loc[b1, "X"]    if b1 in node_lkp.index else None
    n1    = node_lkp.loc[b1, "Y"]    if b1 in node_lkp.index else None

    link_rows.append(dict(
        link_name=row["link"],
        bus0=b0, bus1=b1,
        ss0=row.get("ss0",""), ss1=row.get("ss1",""),
        volts_kv=row.get("volts","350"),
        carrier=row.get("carrier","DC"),
        length_km=row["length"],
        efficiency=row.get("efficiency",""),
        build_year=row.get("build_year",""),
        long0=long0, lat0=lat0, long1=long1, lat1=lat1,
        nztm_e0=e0, nztm_n0=n0, nztm_e1=e1, nztm_n1=n1,
    ))

df_links_out = pd.DataFrame(link_rows)
df_links_out.to_csv(out_dir / "network_links.csv", index=False)
print(f"network_links.csv:  {len(df_links_out)} HVDC link segments")

print("\nAll GIS output files written to:", out_dir)
