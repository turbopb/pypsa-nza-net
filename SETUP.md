# PyPSA-NZA Module Setup Guide

This guide applies to all PyPSA-NZA modules:
- `pypsa-nza-data`
- `pypsa-nza-net`
- `pypsa-nza-heuristic`

---

## 1. Directory Structure

All modules share a common workspace directory for data and model outputs.
The required structure is:

```
<project_root>/                      Set as PYPSA_NZA_ROOT
    pypsa_nza_workspace/
        config/                      YAML configuration files
        data/
            manual/                  Manually compiled static data
            raw/                     Raw source data (EA, Transpower)
            processed/               Pipeline outputs
                static/
                annual/
        models/
            networks/                PyPSA network CSV folders
    pypsa-nza-data/                  Downloaded from GitHub
    pypsa-nza-net/                   Downloaded from GitHub
    pypsa-nza-heuristic/             Downloaded from GitHub
    nza_root.py                      Shared root detection module
```

You only need to download the modules you intend to use.  Each module
operates independently.

---

## 2. Download

Clone or download the required modules from GitHub into your project root:

```bash
cd <project_root>
git clone https://github.com/turbopb/pypsa-nza-data.git
git clone https://github.com/turbopb/pypsa-nza-net.git
git clone https://github.com/turbopb/pypsa-nza-heuristic.git
```

Copy `nza_root.py` (provided with each module) into `<project_root>/`.

---

## 3. Create the Workspace

Create the workspace directory structure:

**Windows (PowerShell):**
```powershell
$root = "C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace"
New-Item -ItemType Directory -Force -Path "$root\config"
New-Item -ItemType Directory -Force -Path "$root\data\manual"
New-Item -ItemType Directory -Force -Path "$root\data\raw\static"
New-Item -ItemType Directory -Force -Path "$root\data\processed\static"
New-Item -ItemType Directory -Force -Path "$root\data\processed\annual"
New-Item -ItemType Directory -Force -Path "$root\models\networks"
```

**Linux (bash):**
```bash
root="/home/<user>/analysis/pypsa_nza_workspace"
mkdir -p "$root"/{config,data/{manual,raw/static,processed/{static,annual}},models/networks}
```

---

## 4. Set the Environment Variable

Set `PYPSA_NZA_ROOT` to your project root directory (the directory that
contains `pypsa_nza_workspace/` and the module repos).  Setting it in
your conda environment means it is available automatically whenever you
activate that environment.

**Windows:**
```powershell
conda activate <your-env>
conda env config vars set PYPSA_NZA_ROOT=C:/Users/Public/Documents/Thesis/analysis/
conda activate <your-env>
```

**Linux:**
```bash
conda activate <your-env>
conda env config vars set PYPSA_NZA_ROOT=/home/<user>/analysis/
conda activate <your-env>
```

Note: you must reactivate the environment after setting a conda env var
for it to take effect.

Verify the variable is set:
```powershell
# Windows
echo $env:PYPSA_NZA_ROOT

# Linux
echo $PYPSA_NZA_ROOT
```

---

## 5. Verify Setup

Run `nza_root.py` directly to confirm the environment is configured
correctly:

```bash
python nza_root.py
```

Expected output:
```
============================================================
PyPSA-NZA Root Configuration
============================================================
Platform  : win32
ROOT_DIR  : C:\Users\Public\Documents\Thesis\analysis
WORKSPACE : C:\Users\Public\Documents\Thesis\analysis\pypsa_nza_workspace
  OK  ...\pypsa_nza_workspace\config
  OK  ...\pypsa_nza_workspace\data
  OK  ...\pypsa_nza_workspace\models
============================================================
```

If any directory shows `MISSING`, create it before running module scripts.

---

## 6. Configuration File

Each module reads a YAML configuration file from:
```
pypsa_nza_workspace/config/nza_net_config.yaml
```

A template configuration file is provided in each module repo under
`config/`.  Copy it to the workspace config directory and edit the
settings as required.  The `paths.root` key in the YAML must match
your `PYPSA_NZA_ROOT` + `pypsa_nza_workspace/`:

```yaml
paths:
    root: "C:/Users/Public/Documents/Thesis/analysis/pypsa_nza_workspace"
```

---

## 7. Run Order (pypsa-nza-net)

```
# Step 1: Prepare data (pypsa-nza-data)
python pypsa-nza-data/build_generator_registry.py

# Step 2: Build base networks
python pypsa-nza-net/nza_base_net.py

# Step 3: Customise networks
python pypsa-nza-net/nza_custom_net.py

# Step 4: Optimise (pypsa-nza-heuristic)
python pypsa-nza-heuristic/nza_cx_opt.py
```

---

## Troubleshooting

**`RuntimeError: PYPSA_NZA_ROOT environment variable is not set`**
The environment variable was not set or the conda environment was not
reactivated after setting it.  Run `conda activate <your-env>` and try
again.

**`RuntimeError: PYPSA_NZA_ROOT ... does not exist`**
The path in the environment variable is wrong or the directory has not
been created.  Check the variable value with `echo $env:PYPSA_NZA_ROOT`
(Windows) or `echo $PYPSA_NZA_ROOT` (Linux).

**`RuntimeError: Workspace directory not found`**
The `pypsa_nza_workspace/` directory does not exist under your project
root.  Follow Step 3 above to create it.
