# pypsa-nza-heuristic: Status, Architecture, and Outstanding Issues
## Document purpose
This document is a handover note for continuing the heuristic capacity planning
work in a new chat session. It describes what the two packages do, what has been
achieved, and exactly what is currently broken and why.

---

## 1. What the two packages do

### pypsa-nza-dispatch
A Python LP dispatch solver package installed as an editable package at:
`C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-dispatch\`

Conda environment: `pypsa-nza-dispatch` (PyPSA 1.2.2 as of June 2026)

**What it does:** Given a PyPSA network CSV folder, it:
1. Loads the network (`load_base_network`)
2. Fixes all capacities to non-extendable (`fix_all_capacities`)
3. Adds load shedding generators at every bus at $10,000/MWh (`add_load_shedding_generators`)
4. Runs PyPSA LP optimisation (`network.optimize()`)
5. Reports how much load was shed — if shed < 0.01% the month is ADEQUATE

Key files:
- `pypsa_nza_dispatch/validate.py` — `run_dispatch()`, `validate_single_scenario()`
- `pypsa_nza_dispatch/network.py` — `load_base_network()`, `fix_all_capacities()`
- `pypsa_nza_dispatch/diagnostics.py` — `calculate_diagnostics()`
- `pypsa_nza_dispatch/utils.py` — `get_network_path()` ← was hardcoded, now reads from config
- `config/dispatch_config.yaml` — points to `cases/reference_v2/` network folders

Networks are read from:
`C:\Users\Public\Documents\Thesis\analysis\dispatch_data\cases\reference_v2\{year}\{month}_{year}\`

---

### pypsa-nza-heuristic
A capacity planning workflow at:
`C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-heuristic\`

**What it does:** Iterative capacity expansion planning across 2024→2030→2035→2040:
1. Takes validated 2024 base network
2. Scales demand to target year using MBIE growth factors
3. Calls pypsa-nza-dispatch to test adequacy for all 12 months
4. User manually specifies capacity additions in a YAML file
5. Applies additions and retests until all 12 months are adequate
6. Repeats for next planning horizon year

Key scripts:
- `02_scenarios/01_setup_year.py` — scales demand from source to target year
- `02_scenarios/03_apply_additions_manual.py` — applies YAML capacity additions
- `03_validation/02_validate_year.py` — runs 12-month adequacy validation
- `02_scenarios/additions/*.yaml` — capacity addition specifications

MBIE demand growth factors (relative to 2024):
- 2030: 1.1382
- 2035: 1.2447
- 2040: 1.3782

Adequacy threshold: load shedding < 0.01% of monthly demand

---

## 2. What we are trying to achieve

### Research scenario
- **Tiwai Point aluminium smelter continues to 2040** (no closure)
- **New scenario: Marsden Point hydrogen electrolyser** (~300-600 MW flexible
  load in Northland, NI) replacing the old Tiwai hydrogen scenario
- Planning horizons: 2024 (baseline), 2030, 2035, 2040
- Goal: identify minimum generation and transmission additions needed to maintain
  system adequacy under MBIE reference demand growth with the Marsden Point
  hydrogen load added

### Why we rebuilt the network
The previous heuristic runs (completed early 2026) used an old 165-bus network
built by a separate pipeline (`nza_cx_net.py`). This network had:
- Incorrect line impedances (some lengths in metres not km)
- Only single-island topology (no proper combined SI+NI)
- Different bus naming conventions

The new validated combined network from `pypsa-nza-net` has:
- 234 buses (238 minus 4 DC intermediate buses)
- 283 AC lines + 1 HVDC Link (BEN→HAY_220)
- Correct impedances (validated via DC power flow)
- Both islands connected via HVDC

---

## 3. What works

### pypsa-nza-net (validation — COMPLETE)
- SI network: 81 buses, 95 lines, validated 24 months, max loading 100% Jul 2024 (S-37) ✓
- NI network: 157 buses, 188 lines, validated 24 months, max loading 65% Jul 2024 (N-87) ✓
- Combined network: 234 buses, 284 lines, validated 24 months, 0 overloads ✓
- Theta range Jan 2024: [-0.37, +0.09] rad, max line angle diff 6.4° ✓
- Three NI lines corrected (lengths were in metres not km): N-88, N-107, N-188 ✓

### Network build pipeline (pypsa-nza-net)
- `nza_base_net.py` builds SI, NI, combined networks for all 24 months ✓
- `nza_custom_net.py` customises SI networks only (p_max_pu, costs) ✓
- `fix_combined_p_max_pu.py` copies p_max_pu from SI/NI into combined networks ✓
- `build_dispatch_v2.py` builds clean dispatch networks:
  - Removes 3-segment HVDC chain and DC buses
  - Adds single HVDC Link BEN→HAY_220 (p_nom=1200 MW, eta=0.965)
  - Sets p_max_pu: hydro=0.75, geothermal=0.87, all others=1.00 ✓
- `verify_dispatch_v2.py` verifies 12 months: 234 buses, 283 lines, 1 link, 48 gens ✓

### Dispatch network contents (reference_v2/2024/)
All 12 months built and verified:
- 234 buses, 283 AC lines, 1 HVDC Link, 48 generators, ~145 loads
- Total available generation: 6485 MW
- Mean January load: 2080 MW (adequate headroom)
- p_max_pu correct: hydro 0.75, geothermal 0.87, rest 1.00
- HVDC Link: BEN→HAY_220, p_nom=1200 MW, efficiency=0.965, NO carrier set

---

## 4. RESOLVED: Dispatch solver now working

### Status: 12/12 adequate months, 0 load shedding (June 2026)

### Root cause (identified and fixed)
Two problems were found and fixed in `build_dispatch_v2.py`:

**Problem 1: s_nom=0 on all lines**
PyPSA stores line thermal ratings via `type` strings (e.g. "490-AL1/64-ST1A 220.0")
and calculates `s_nom` at runtime via `calculate_dependent_values()`. This value
is never written back to `lines.csv`. When the dispatch solver loads the network,
all lines have `s_nom=0`, meaning no power can flow on any line. The solver can
only dispatch generators directly connected to load buses (~68 MW of local NI
generation), leaving all SI generation stranded.

**Fix:** Read `s_nom_MW` from `line_loading.csv` produced by the DC power flow
validation runs (in `verification_pf/combined/2024/{month}_2024/line_loading.csv`)
and write it explicitly into the network before saving.

**Problem 2: Integer index in loads-p_set.csv**
PyPSA 1.0.3 writes `loads-p_set.csv` with an integer index (0, 1, 2...).
PyPSA 1.2.2 requires a datetime index matching the network snapshots. With an
integer index, all loads default to `p_set=0`, giving zero demand.

**Fix:** After `export_to_csv_folder()`, rewrite `loads-p_set.csv` with
`n.snapshots` as the index.

### What did NOT work (for reference)
- Setting `carrier='DC'` on HVDC link
- Zeroing line reactances (x_pu_eff=0) at runtime
- Removing KVL constraint via extra_functionality
- Upgrading PyPSA 1.0.7 → 1.2.2
All these addressed symptoms not the root cause (s_nom=0 and integer index).

---

## 5. Next steps

### Step 1 — COMPLETE: 2024 baseline validated
Result: 12/12 adequate months, 0 load shedding, max line loading 84.9%.

### Step 2 — Setup and validate 2030
```powershell
cd C:\Users\Public\Documents\Thesis\analysis\pypsa-nza-heuristic
conda activate pypsa-nza-dispatch
python 02_scenarios\01_setup_year.py --source-year 2024 --target-year 2030
python 03_validation\02_validate_year.py --year 2030
```
Expected: some inadequate months due to demand growth (factor 1.1382).
Identify bottleneck type (generation or transmission) and add capacity via YAML.

### Step 3 — Iterate capacity additions for 2030
Edit `02_scenarios/additions/additions_2030_iter01.yaml` with generation/
transmission additions. Run:
```powershell
python 02_scenarios\03_apply_additions_manual.py --year 2030 --additions 02_scenarios\additions\additions_2030_iter01.yaml --iteration 1
```
Repeat until 12/12 adequate.

### Step 4 — Repeat for 2035 and 2040
Same process: setup_year → validate → add capacity → revalidate.

### Step 5 — Add Marsden Point hydrogen scenario
Add large flexible electrolyser load (~300-600 MW) at MPE bus (Maungatapere,
Northland). This is on the N-188 corridor (MPE→KOE) which was one of the
corrected lines. Expect this to stress the Northland corridor significantly.

### Complete run sequence
```powershell
# Build dispatch networks (pypsa-nza-data env, run from pypsa-nza-net)
python build_dispatch_v2.py

# All planning steps (pypsa-nza-dispatch env, run from pypsa-nza-heuristic)
python 02_scenarios\01_setup_year.py --source-year 2024 --target-year 2030
python 03_validation\02_validate_year.py --year 2030
python 02_scenarios\03_apply_additions_manual.py --year 2030 --additions 02_scenarios\additions\additions_2030_iter01.yaml --iteration 1
# ... repeat until adequate
python 02_scenarios\01_setup_year.py --source-year 2030 --target-year 2035
python 03_validation\02_validate_year.py --year 2035
# ... and so on to 2040
```

---

## 6. File locations summary

| File | Location |
|---|---|
| Validated combined networks | `pypsa_nza_workspace/models/networks/reference/2024/` |
| Dispatch networks (new) | `dispatch_data/cases/reference_v2/2024/` |
| Dispatch config | `pypsa-nza-dispatch/config/dispatch_config.yaml` |
| utils.py (fixed) | `pypsa-nza-dispatch/pypsa_nza_dispatch/utils.py` |
| validate.py (modified) | `pypsa-nza-dispatch/pypsa_nza_dispatch/validate.py` |
| build_dispatch_v2.py | `pypsa-nza-net/build_dispatch_v2.py` |
| verify_dispatch_v2.py | `pypsa-nza-net/verify_dispatch_v2.py` |
| fix_combined_p_max_pu.py | `pypsa-nza-net/fix_combined_p_max_pu.py` |

