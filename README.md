# pypsa-nza-net

Network build, validation, and DC power flow verification for the New Zealand
electricity transmission system using [PyPSA](https://pypsa.org).

This repository constructs monthly PyPSA network objects for the South Island
(SI) transmission system, validates them against measured 2024 injection and
load data, and performs linearised DC power flow analysis across all modelled
months. The NI network and combined NI+SI network are planned extensions.

---

## Repository Structure

```
pypsa-nza-net/
│
│   CORE PIPELINE
├── nza_base_net.py              Build monthly SI networks from line/node data
├── nza_custom_net.py            Apply customisations (ratings, type fixes)
├── dc_power_flow.py             Validated numpy DC power flow (thesis figures)
├── nza_net_pf.py                Batch DC power flow across all months
│
│   PLOTTING
├── plot_dc_validation.py        Thesis validation figures (Jan/Jul 2024)
├── plot_pf_summary.py           24-month heatmap + S-37 summary figure
├── plot_network_topology.py     Network topology map
│
│   UTILITIES
├── compute_line_lengths.py      Arc-distance line length calculator
├── generate_network_gis_files.py  GIS export for QGIS
├── generate_node_sets.py        Node set generator
├── nztm_geod.py                 NZTM2000 geodesy module (imported by others)
├── nztm_coords_to_wgs84.py      Coordinate conversion utility
├── validate_lines_nodes.py      Data consistency checks
│
│   DOCUMENTATION
├── README.md                    This file
├── SETUP.md                     Environment setup instructions
├── LICENSE
│
└── archive/                     Superseded scripts (retained for reference)
    ├── nza_net_dc.py            Earlier numpy DC PF (superseded by nza_net_pf.py)
    ├── verify_power_flow.py     Attempted n.pf() AC flow (diverges on SI network)
    └── diagnostics/             Debugging artefacts from network build phase
```

---

## Prerequisites

See `SETUP.md` for full environment setup. In summary:

- Python 3.11, conda environment `pypsa-nza-data`
- PyPSA 1.0.3 (tested; 1.2.0 also compatible)
- numpy, pandas, matplotlib, scipy, geopandas
- Environment variable `PYPSA_NZA_ROOT` pointing to the thesis analysis root,
  e.g. `C:\Users\Public\Documents\Thesis\analysis`

The workspace is expected at `$PYPSA_NZA_ROOT/pypsa_nza_workspace/` with the
following subdirectory layout:

```
pypsa_nza_workspace/
├── data/
│   ├── manual/
│   │   ├── lines_data_si.csv        Transmission line parameters (authority)
│   │   └── nodes.csv                Node register (island classification only)
│   └── processed/
│       └── annual/
│           ├── 2024_import_sites_all.csv   GIP timeseries (both islands)
│           └── 2025_import_sites_all.csv
├── models/
│   └── networks/
│       └── reference/
│           ├── 2024/
│           │   ├── si_jan_2024/     PyPSA CSV network folders
│           │   └── ...
│           └── 2025/
└── outputs/
    └── verification_pf/             DC power flow results (written by pipeline)
```

---

## Pipeline: South Island Network

Run scripts in the following order. All scripts are run from the
`pypsa-nza-net` directory with the conda environment activated.

### Step 1 — Build monthly SI networks

```powershell
python nza_base_net.py --year 2024
python nza_base_net.py --year 2025
```

Builds 12 monthly PyPSA networks per year from `lines_data_si.csv` and the
measured load/generation timeseries. Networks are saved as PyPSA CSV folders
under `models/networks/reference/{year}/si_{month}_{year}/`.

**Key conventions:**
- Topology authority is `lines_data_si.csv` (81 buses: 72 line endpoints +
  9 multi-voltage splits). `nodes.csv` is used only for island classification.
- AC buses only. DC buses (BEN-DC, FTB) are present in the saved network but
  are removed at power flow time.
- SI has 29 GIP sites in the measured timeseries. HVDC export is loaded
  separately from the `hvdc/` processed data folder.
- Loads are filtered to 57 SI GXP sites using the `island` column in
  `nodes.csv`.

**Known bugs fixed (do not revert):**
1. Load island filtering — export timeseries filtered to SI sites before
   aggregation. Correct SI mean load ~864 MW (Jan 2024).
2. Ghost buses — topology restricted to `lines_data_si.csv` endpoints only.
   Result: 81 buses (was incorrectly 111).
3. DC bus contamination — nearest-bus aggregation restricted to AC buses.
   Prevents FTB absorbing Canterbury loads due to geographic proximity.
4. Datetime index — `loads-p_set.csv` rewritten after export with correct
   datetime index (PyPSA 1.0.3 writes integer index by default).

### Step 2 — Apply network customisations

```powershell
python nza_custom_net.py --year 2024
python nza_custom_net.py --year 2025
```

Applies post-build customisations: corrected line ratings (notably S-L3
INV→TWI), custom transformer types (`630MVA 220/110kV`), and any manual
overrides. Modifies the networks in-place.

**Note:** `register_custom_transformer_types()` must be called before
`calculate_dependent_values()`. This ordering is enforced in the script.

### Step 3a — Validated DC power flow (January and July 2024)

```powershell
python dc_power_flow.py --year 2024 --month 1
python dc_power_flow.py --year 2024 --month 7
```

Solves B·θ=P directly with numpy for a single month. This script produced the
thesis validation results. Outputs written to
`outputs/verification/{year}/{month}/`.

**Validated results:**
- Jan 2024: SI load 864 MW, generation 986 MW, HVDC 293 MW. No overloads.
  S-37 (ISL–KIK 220 kV) max loading 79.6%.
- Jul 2024: S-37 at 99% (binding winter constraint). S-14 (ROX–ISL) at 61%.

### Step 3b — Batch DC power flow across all months

```powershell
for ($m = 1; $m -le 12; $m++) { python nza_net_pf.py --year 2024 --month $m }
for ($m = 1; $m -le 12; $m++) { python nza_net_pf.py --year 2025 --month $m }
```

Runs the same numpy B·θ=P solver for all 24 months. Outputs per-month
`line_loading.csv` and `power_flow_verification.png` under
`outputs/verification_pf/{year}/{month}/`.

**Implementation note:** PyPSA 1.x `n.lpf()` was evaluated and found
unsuitable for standalone use — the preparation phase overwrites `buses_t.p`
before the solve, discarding externally-set injections. The numpy solver
bypasses this entirely and is version-agnostic.

### Step 4 — Generate summary figures

```powershell
python plot_dc_validation.py
```

Produces thesis validation figures:
- `si_network_loading_jan2024.png`
- `si_network_loading_jul2024.png`
- `si_line_loading_comparison.png`
- `si_monthly_balance_2024.png`

```powershell
python plot_pf_summary.py [--top-n 20]
```

Produces cross-month summary figure (`pf_summary.png`):
- Top panel: heatmap of top-N most loaded lines across all 24 months
- Bottom panel: S-37 monthly maximum loading with seasonal annotation

---

## Key Network Facts

| Item | Value |
|---|---|
| Buses (SI) | 81 (72 line endpoints + 9 multi-voltage splits) |
| Lines (SI) | 95 |
| Transformers (SI) | 9 |
| Generators (SI, modelled) | 12 |
| Loads (SI) | 44 |
| GIP sites (measured timeseries) | 29 |
| GXP sites (SI load) | 57 |
| Reference year | 2024 |
| Modelled months | Jan 2024 – Dec 2025 (24 months) |
| Slack bus | BEN (Benmore 220 kV) |
| Binding winter constraint | S-37 (ISL–KIK 220 kV), max 100.1% in Jul 2024 |

---

## HVDC Topology

```
BEN-DC  →  FTB  →  [undersea cable]  →  OTB  →  HAY-DC
(SI converter,   (cable landing,              (NI converter,
 Benmore)         near Blenheim,               Haywards)
                  not a converter)
```

In the SI-only model, HVDC export is represented as a negative load at BEN.
BEN-DC and FTB buses are removed before power flow.

---

## PyPSA API Notes

These findings apply to PyPSA 1.0.3 and are retained here to avoid
re-discovering them.

- `n.lpf(skip_pre=True)` — raises `AttributeError: SubNetwork has no
  attribute p_bus_shift`. The preparation phase must run to set this
  attribute, but it also overwrites `buses_t.p`.
- `n.lpf()` — preparation phase overwrites `buses_t.p` with internally
  computed (empty) injections, producing NaN bus angles and NaN line flows.
- `n.pf()` — full Newton-Raphson AC flow. Diverges on the SI network due to
  54–67° voltage angle differences across transformers.
- `register_custom_transformer_types()` must precede
  `calculate_dependent_values()`, otherwise non-standard transformer types
  are not found and `x_pu` is set to NaN.
- `x_pu_eff` = `x_pu / num_parallel`. Use `x_pu_eff` when building the
  B matrix; it already incorporates parallel circuits.
- `loads-p_set.csv` written by `export_to_csv_folder()` uses integer row
  index. Rewrite immediately after export using `network.snapshots` as index.

---

## Planned Extensions

- NI network build and validation (same pipeline as SI)
- Combined NI+SI network with HVDC link (BEN-DC → HAY-DC)
- Heuristic capacity expansion (pypsa-nza-heuristic)
