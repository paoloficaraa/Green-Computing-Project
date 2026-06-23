# Green-computing_final-project

Green Computing project. The primary focus is on energy and computational efficiency rather than model accuracy.

## Structure

```
├── datasets/                        # Bio/health informatics CSV datasets
│   ├── 10_7717_peerj_5665_dataYM2018_neuroblastoma.csv
│   ├── dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv
│   ├── dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv
│   ├── journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv
│   └── journal.pone.0158570_S2File_depression_heart_failure.csv
├── script1.py                       # Baseline: RandomForest (100 trees, 100 splits)
├── script2.py                       # Optimized: RandomForest (50 trees, 50% sample, n_jobs=-1)
├── script3/                         # Rust-based experiment (requires Rust toolchain)
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── src/
│   ├── target/
│   └── rust_tracker.py              # Python wrapper that runs the Rust binary + tracks emissions
├── requirements.txt
├── CodeCarbon reports/              # Auto-generated emissions CSV (from CodeCarbon)
└── mcc reports/                     # Auto-generated MCC results CSV
```

## Prerequisites

- Python 3.9+
- `pip install -r requirements.txt`  (pandas, numpy, scikit-learn, codecarbon)
- Rust toolchain *(optional, only for script3)* — install via `rustup`
- GPU driver with NVML access installed for NVML to work

## Usage

### 1. Baseline Experiment (`script1.py`)

Runs a standard `RandomForestClassifier` (100 estimators, 100 stratified train/test splits) on each dataset and records emissions + MCC scores.

```bash
python script1.py
```

Output:
- `CodeCarbon reports/emissions_script1.csv` — carbon/energy report
- `mcc reports/mcc_report_script1.csv` — average MCC per dataset

### 2. Optimized Experiment (`script2.py`)

Same as script1 but with optimizations: 50% sample size, 50 estimators, `n_jobs=-1` (all cores).

```bash
python script2.py
```

Output:
- `CodeCarbon reports/emissions_script2.csv`
- `mcc reports/mcc_report_script2.csv`

### 3. Rust-based Experiment (`script3/`)

Compiles and runs a Rust binary (using `polars` + `smartcore`) while tracking emissions from Python.

```bash
# Build the Rust binary
cd script3
cargo build --release

# Run the tracker (measures the compiled binary)
python rust_tracker.py
```

Output:
- `CodeCarbon reports/emissions_script3.csv`
- `mcc reports/mcc_report_script3.csv`

## Comparing Results

Compare the generated CSV files in `CodeCarbon reports/` and `mcc reports/` to evaluate the trade-off between computational cost and model performance across scripts.
