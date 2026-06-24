# Green Computing: Energy-Efficient Machine Learning

Green Computing project. The primary focus is on energy and computational efficiency rather than model accuracy.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [1. Baseline Experiment](#1-baseline-experiment-script1py)
  - [2. Optimized Experiment](#2-optimized-experiment-script2py)
  - [3. Julia-based Experiment](#3-julia-based-experiment-script3jl)
- [Comparing Results](#comparing-results)
- [Contributors](#contributors)

---

## Features

- Energy and carbon emission tracking via [CodeCarbon](https://mlco2.github.io/codecarbon/)
- Benchmarking across bio/health informatics datasets
- Comparison of Python (scikit-learn) vs. Julia (MLJ + ScikitLearn interface) implementations
- Reporting of both computational cost and model performance (MCC score)

---

## Project Structure

```
├── datasets/                                  # Bio/health informatics CSV datasets
│   ├── 10_7717_peerj_5665_dataYM2018_neuroblastoma.csv
│   ├── dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv
│   ├── dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv
│   ├── journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv
│   └── journal.pone.0158570_S2File_depression_heart_failure.csv
├── script1.py                                 # Baseline: RandomForest (100 trees, 100 splits)
├── script2.py                                 # Optimized: RandomForest (50 trees, 50% sample, n_jobs=-1)
├── script3.jl                                 # Julia experiment (MLJ + ScikitLearn interface)
├── requirements.txt                           # Python dependencies
├── CodeCarbon reports/                        # Auto-generated emissions CSV (from CodeCarbon)
└── mcc reports/                               # Auto-generated MCC results CSV
```

---

## Prerequisites

- **Python 3.9+**
- **Julia** (optional, only for `script3.jl`)
- **Python packages:** `pip install -r requirements.txt` (pandas, numpy, scikit-learn, codecarbon)
- **GPU driver with NVML access** (for GPU power monitoring; falls back to CPU if unavailable)

---

## Usage

### 1. Baseline Experiment (`script1.py`)

Runs a standard `RandomForestClassifier` (100 estimators, 100 stratified train/test splits) on each dataset and records emissions + MCC scores.

```bash
python script1.py
```

**Output:**
- `CodeCarbon reports/emissions_script1.csv` — carbon/energy report
- `mcc reports/mcc_report_script1.csv` — average MCC per dataset

---

### 2. Optimized Experiment (`script2.py`)

Same as script1 but with optimizations: 50% sample size, 50 estimators, `n_jobs=-1` (all cores).

```bash
python script2.py
```

**Output:**
- `CodeCarbon reports/emissions_script2.csv`
- `mcc reports/mcc_report_script2.csv`

---

### 3. Julia-based Experiment (`script3.jl`)

Runs a Julia-based pipeline (using `MLJ` + `MLJScikitLearnInterface`) that replicates the optimized experiment in Julia while tracking emissions via Python's CodeCarbon through `PythonCall.jl`.

```bash
julia script3.jl
```

**Output:**
- `CodeCarbon reports/emissions_script3.csv`
- `mcc reports/mcc_report_script3.csv`

---

## Comparing Results

After running the experiments, compare the generated CSV files in `CodeCarbon reports/` and `mcc reports/` to evaluate the trade-off between computational cost and model performance across scripts.

- **Energy:** Check `CodeCarbon reports/emissions_script*.csv` for total energy consumption (kWh) and CO₂ emissions.
- **Performance:** Check `mcc reports/mcc_report_script*.csv` for the average MCC score per dataset.
