# pypsa-nza-net

Network build, validation, and DC power flow verification for the New Zealand
electricity transmission system using [PyPSA](https://pypsa.org).

Constructs monthly PyPSA network objects for the South Island (SI) and North
Island (NI) transmission systems, validates them against measured 2024
injection and load data, and performs linearised DC power flow analysis across
all modelled months. The combined SI+NI network with HVDC link is the next
planned step.

---

## Repository Structure

```
pypsa-nza-net/
│
│   CORE PIPELINE
├── nza_base_net.py              Build monthly SI and NI networks from line/node data
├── nza_custom_net.py            Apply customisations (ratings, type fixes)
├── dc_power_flow.py             Validated numpy DC power flow -- SI thesis figures
├── nza_net_pf.py                Batch DC power flow across all months (SI and NI)
│
│   PLOTTING
├── plot_dc_validation.py        SI thesis validation figures (bar chart, monthly balance)
├── plot_si_network_loading.py   SI network loading maps -- Jan and Jul 2024
├── plot_ni_network_loading.py   NI network loading maps -- Jan and Jul 2024
├── plot_ni_line_loading.py      NI line loading bar chart -- Jan vs Jul 2024
├── plot_pf_summary.py           24-month heatmap + binding constraint bar chart
├── plot_network_topology.py     Network topology map (buses, lines, transformers)
│
│   UTILITIES
├── compute_line_lengths.py      Arc-distance line length calculator
├── generate_network_gis_files.py  GIS export for QGIS
├── generate_node_sets.py        Node set generator
├── nztm_geod.py                 NZTM2000 geodesy module (imported by other scripts)
├── nztm_coords_to_wgs84.py      Coordinate conversion utility
├── validate_lines_nodes.py      Data consistency checks
│
│   DOCUMENTATION
├── README.md                    This file
├── SETUP.md                     Environment setup instructions
├── LICENSE
│
└── archive/                     Superseded scripts (retained for reference)
```

---

## Prerequisites

See `SETUP.md` for full environment setup. In summary:

- Python 3.11, conda environment `pypsa-nza-data`
- PyPSA 1.0.3 (tested; 1.2.x also compatible)
- numpy, pandas, matplotlib, scipy, cartopy
- Environment variable `PYPSA_NZA_ROOT` pointing to the thesis analysis root,
  e.g. `C:\Users\Public\Documents\Thesis\analysis`

The workspace is expected at `$PYPSA_NZA_ROOT/pypsa_nza_workspace/` with the
following subdirectory layout:

```
pypsa_nza_workspace/
├── data/
│   ├── manual/
│   │   ├── nodes.csv                Master site register (island classification,
│   │   │                            coordinates). JCT_ prefix = manually added
│   │   │                            junction points digitised in QGIS.
│   │   ├── lines_data_si.csv        SI transmission line parameters (topology authority)
│   │   └── lines_data_ni.csv        NI transmission line parameters (topology authority)
│   └── processed/annual/
│       ├── {year}_export_sites_all.csv   GXP load timeseries (all NZ)
│       └── {year}_import_sites_all.csv   GIP generation timeseries (all NZ)
├── models/networks/reference/
│   ├── 2024/
│   │   ├── si_jan_2024/     SI PyPSA CSV network folders (12 months)
│   │   ├── ni_jan_2024/     NI PyPSA CSV network folders (12 months)
│   │   └── ...
│   └── 2025/
└── outputs/
    ├── figures/validation/  Thesis validation figures
    ├── figures/network/     Topology figures
    └── verification_pf/     DC power flow results
        ├── {year}/          SI results
        └── NI/{year}/       NI results
```

---

## Pipeline: South Island Network

### Step 1 -- Build monthly SI networks
```powershell
python nza_base_net.py --year 2024
python nza_base_net.py --year 2025
```

### Step 2 -- Apply customisations
```powershell
python nza_custom_net.py --year 2024
python nza_custom_net.py --year 2025
```

### Step 3 -- Run DC power flow (all months)
```powershell
for ($m = 1; $m -le 12; $m++) { python nza_net_pf.py --year 2024 --month $m --island SI }
for ($m = 1; $m -le 12; $m++) { python nza_net_pf.py --year 2025 --month $m --island SI }
```

### Step 4 -- Generate SI figures
```powershell
python plot_dc_validation.py
python plot_si_network_loading.py
python plot_pf_summary.py --island SI
```

---

## Pipeline: North Island Network

### Step 1 -- Build monthly NI networks
```powershell
python nza_base_net.py --year 2024
python nza_base_net.py --year 2025
```
NI networks are built automatically when `lines_data_ni.csv` exists.

### Step 2 -- Apply customisations
```powershell
python nza_custom_net.py --year 2024
python nza_custom_net.py --year 2025
```

### Step 3 -- Run DC power flow (all months)
```powershell
for ($m = 1; $m -le 12; $m++) { python nza_net_pf.py --year 2024 --month $m --island NI }
for ($m = 1; $m -le 12; $m++) { python nza_net_pf.py --year 2025 --month $m --island NI }
```

### Step 4 -- Generate NI figures
```powershell
python plot_ni_network_loading.py
python plot_ni_line_loading.py
python plot_pf_summary.py --island NI
```

---

## Key Network Facts

| Item | SI | NI |
|---|---|---|
| Buses | 81 | 157 |
| Lines | 95 | 188 |
| Transformers | 9 | 16 |
| Generators (modelled) | 12 | 36 |
| Loads | 44 | ~101 |
| GIP sites (measured) | 29 | 54 |
| GXP sites (load) | 57 | ~102 |
| Slack bus | BEN (Benmore 220 kV) | HAY_110 (Haywards, HVDC terminal) |
| Binding winter constraint | S-37 (ISL-KIK 220 kV), max 100% Jul 2024 | N-87 (HAY-TKR 110 kV), max 65% Jul 2024 |

---

## HVDC Topology

```
BEN-DC  ->  FTB  ->  [Cook Strait cable]  ->  OTB  ->  HAY-DC
(SI converter,   (cable landing,                    (NI converter,
 Benmore)         Fighting Bay)                      Haywards)
```

In standalone SI model: HVDC export is a negative load at BEN. BEN-DC and FTB
removed before power flow.

In standalone NI model: HVDC import is represented by a slack generator at
HAY_110. HAY-DC and OTB removed before power flow.

In the combined model (planned): HVDC modelled as a PyPSA Link with resistance.

---

## PyPSA API Notes (version 1.0.3)

- `n.lpf()` -- preparation phase overwrites `buses_t.p`, producing NaN angles.
  All power flow uses direct numpy B*theta=P solve instead.
- `n.pf()` -- full Newton-Raphson AC flow. Diverges on large angle differences.
  Not used.
- `register_custom_transformer_types()` must precede `calculate_dependent_values()`.
- Bus voltage is assigned from `lines_data` endpoints, not from `nodes.csv`.
  The `volts` column in `nodes.csv` is the distribution connection voltage,
  not the transmission bus voltage.
- `loads-p_set.csv` written by `export_to_csv_folder()` uses integer row index.
  Rewritten with datetime index in `save_network()`.

---

## Planned Extensions

1. Combined SI+NI network with Cook Strait HVDC link (BEN-DC -> HAY-DC)
2. Heuristic capacity expansion 2025-2040 (pypsa-nza-heuristic)
