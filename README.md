# Green Computing: Energy-Efficient Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comparative analysis of energy consumption for Random Forest classification across **Python (scikit-learn)** and **Julia (native DecisionTree.jl)** implementations, measured via hardware-level CPU energy counters on Windows 11.

## Table of Contents

- [Project Overview](#project-overview)
- [Three Experiments](#three-experiments)
  - [1. Baseline (script1.py)](#1-baseline-script1py)
  - [2. Optimized Python (script2.py)](#2-optimized-python-script2py)
  - [3. Julia Native (script3/)](#3-julia-native-script3)
- [Measurement Methodology (Key Learning)](#measurement-methodology-key-learning)
  - [How CodeCarbon Measures Energy](#how-codecarbon-measures-energy)
  - [The Three Tracking Modes](#the-three-tracking-modes)
  - [The EMI Bug: What I Discovered](#the-emi-bug-what-i-discovered)
  - [Diagnosing the Problem](#diagnosing-the-problem)
  - [The Fix: windows_support Branch](#the-fix-windows_support-branch)
  - [Before vs. After EMI Comparison](#before-vs-after-emi-comparison)
  - [Why Joblib Multiprocessing Was Under-Reported](#why-joblib-multiprocessing-was-under-reported)
  - [External vs. Internal Tracking](#external-vs-internal-tracking)
- [Results](#results)
  - [Final Comparison Table](#final-comparison-table)
  - [Visual Comparison of CPU Power Readings](#visual-comparison-of-cpu-power-readings)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
  - [Prerequisites](#prerequisites)
  - [Installation: CodeCarbon from windows_support](#installation-codecarbon-from-windows_support)
  - [Installation: Julia Environment](#installation-julia-environment)
  - [Run All Experiments](#run-all-experiments)
  - [Generate Comparison](#generate-comparison)
- [Known Pitfalls & Lessons Learned](#known-pitfalls--lessons-learned)
- [License](#license)

---

## Project Overview

This project compares the energy consumption and CO₂ emissions of three implementations of the same machine learning task (Random Forest classification on 5 bio/health informatics datasets) using **CodeCarbon** for measurement. The core research question:

> **How much energy can be saved while preserving model quality by targeting the computational cost inside each tree and forest sizing, rather than by relying on aggressive row subsampling?**

The final strategy combines 60 regularized trees (`n_estimators=60`, `max_depth=12`, `min_samples_leaf=5`, `min_samples_split=10`), variance-based feature selection (top 60%, floor 8), and single-threaded Python execution (eliminating joblib multiprocessing overhead on small tabular datasets). This achieves a **42.88% energy reduction in Python** and **47.07% in Julia** while preserving an average MCC of **0.595 / 0.590** (against 0.599 baseline).

Each experiment runs 100 stratified train/test splits on each of 5 datasets, and measures:

- **Energy consumption** (in Watt-hours, Wh)
- **CO₂ emissions** (in grams CO₂-eq)
- **Execution time** (in seconds)
- **Model quality** via Matthews Correlation Coefficient (MCC)

---

## Three Experiments

### 1. Baseline (`script1.py`)

Standard Python implementation:

- `RandomForestClassifier(n_estimators=100, random_state=42)`
- Full dataset (100% rows and features)
- Single-threaded (no `n_jobs` parameter)
- CodeCarbon **internal** tracking

### 2. Optimized Python (`script2.py`)

Optimized Python implementation (Variant E):

- `RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1)`
- `max_depth=12`
- `min_samples_leaf=5`
- `min_samples_split=10`
- variance-based feature selection before training, keeping the top-ranked features up to roughly 60% of the original dimensionality with a minimum of 8 features
- single-threaded execution (`n_jobs=1`) to avoid joblib process-pool spawn and IPC overhead on small EHR datasets
- no row subsampling (preserves full clinical training distribution)
- CodeCarbon **internal** tracking

### 3. Julia Native (`script3/`)

Pure Julia implementation (no PythonCall bridge):

- `RandomForestClassifier(n_trees=60)` from `DecisionTree.jl` via `MLJDecisionTreeInterface`
- identical regularized configuration: `max_depth=12`, `min_samples_leaf=5`, `min_samples_split=10`
- identical variance-based feature filtering (top 60%, min 8)
- `Threads.@threads` for native shared-memory parallel execution across 100 splits
- Launched with `julia -t auto` to use all cores
- CodeCarbon **external** tracking via `run_script3.py`
## Measurement Methodology (Key Learning)

This section documents the most important technical discovery of this project: **how CodeCarbon measures energy on Windows, and why the default method is unreliable for multiprocessing workloads.**

### How CodeCarbon Measures Energy

CodeCarbon computes energy consumption as:

```text
Energy = CPU_Power × Duration + GPU_Power × Duration + RAM_Power × Duration
```

The critical question is: **how is CPU_Power determined?**

CodeCarbon queries the CPU's power consumption **every second** during execution and averages the samples. If it cannot get a hardware reading, it falls back to estimation:

```text
CPU_Power_estimated = TDP × (cpu_utilization_percent / 100)
```

Where:

- **TDP** (Thermal Design Power) is a constant — the chip's maximum rated thermal output
- **cpu_utilization_percent** is polled from `psutil.cpu_percent()` every second

### The Three Tracking Modes

CodeCarbon has three CPU tracking methods, in order of preference:

| # | Method | Source | Accuracy | Platforms |
| --- | --- | --- | --- | --- |
| 1 | **RAPL** (Running Average Power Limit) | Hardware energy counters via `perf_event_open()` (Linux) or `IntelPowerGadget` (macOS) | **Very High** (real hardware registers) | Linux, macOS |
| 2 | **Windows EMI** (Energy Meter Interface) | Hardware energy counters via Windows Power Management API | **Very High** (same RAPL registers, different OS API) | Windows only |
| 3 | **CPU Load Estimation** | `TDP × cpu_utilization_percent` | **Low** (estimated, not measured) | Fallback on all OS |

### The EMI Bug: What I Discovered

**The problem:** CodeCarbon v3.2.8 (the latest stable release on PyPI) has Windows EMI support **disabled by default** in its tracking mode selection logic. When running on Windows, instead of trying EMI, it falls through to the estimation method. This is a known issue tracked in [CodeCarbon PR #1263](https://github.com/mlco2/codecarbon/pull/1263) ("Tracking on Windows: add Windows Energy Meter Interface").

**Why it exists:** The EMI integration was contributed as a pull request but was never merged into a release. The EMI code is present in the repository but is gated behind conditionals that never activate in v3.2.8.

**The result:** On Windows with the standard `pip install codecarbon`, every measurement uses `TDP × cpu_load` estimation, which is unreliable.

### Diagnosing the Problem

I discovered the issue during result analysis. The initial measurements showed:

```text
# OLD (before EMI fix) — suspicious results
Script          CPU_Power   CPU_Util    Duration   Energy
Baseline        10.97W      ~8%         72.0s      0.960 Wh
Optimized       11.70W      ~0%         46.1s      0.648 Wh
Julia           58.93W      ~44%        35.7s      0.959 Wh
```

The scripts should have similar CPU power (all running on the same i7-9700K processor). Why was Python showing ~11W while Julia showed ~59W?

**Root cause:** The `cpu_utilization_percent` polling loop in CodeCarbon was missing the joblib child processes:

| Script | Parallel Model | Process Visibility to psutil | Reported CPU Util | Estimated Power |
| --- | --- | --- | --- | --- |
| Baseline (no n_jobs) | None (single-threaded) | Main thread only | ~8% | ~11W (TDP=95W × 8%) |
| Optimized (n_jobs=-1) | joblib multiprocessing | Child processes not polled | ~0% | ~11W (effectively idle power) |
| Julia (Threads.@threads) | OS threads in same process | Same process, all threads | ~44% | ~59W |

**Julia was being measured correctly** because its threads live in the same process as the main Julia process, so `psutil.cpu_percent()` captured all of them. But scikit-learn's `n_jobs=-1` spawns separate child processes via `joblib.Parallel`, and the sampling loop was not aggregating their CPU utilization.

**The comparison was fundamentally unfair:** Optimized Python was being charged idle power (~11W, essentially the system's background power) while Julia was being charged the full computing power (~59W).

### The Fix: `windows_support` Branch

To get accurate hardware measurements on Windows, I installed CodeCarbon from the `windows_support` branch of the official repository:

```bash
pip install git+https://github.com/mlco2/codecarbon.git@windows_support
```

This branch contains the EMI implementation from PR #1263 and properly activates Windows Energy Meter Interface (EMI) as the tracking method instead of the estimation fallback.

**Verification:** After installation, CodeCarbon reports:

```text
CPU Tracking Method: Windows EMI
```

And queries the RAPL energy counters directly via the Windows Power Management API — the same hardware registers that Linux reads via `/sys/class/powercap`.

### Before vs. After EMI Comparison

| Metric | Without EMI (v3.2.8) | With EMI (RAPL hardware) | Δ |
| --- | --- | --- | --- |
| **CPU Tracking Method** | `cpu_load` (estimation) | `Windows EMI` (hardware) | — |
| **Baseline CPU Power** | 10.97 W | 54.0 W | **+392%** |
| **Optimized CPU Power** | 11.70 W | 61.4 W | **+425%** |
| **Julia CPU Power** | 58.93 W | 62.7 W | +6% |

| Script | Energy (old) | Energy (EMI) | Old Conclusion | Correct Conclusion |
| --- | --- | --- | --- | --- |
| Baseline | 0.960 Wh | 1.738 Wh | — (baseline) | — (baseline) |
| Optimized | 0.648 Wh | 1.127 Wh | −32.5% energy (seems good) | **−35.2%** energy (still good!) |
| Julia | 0.959 Wh | 1.013 Wh | −0.1% energy (same as baseline!) | **−41.7%** energy (clear winner!) |

**Key insights:**

1. The old method **under-estimated ALL Python scripts**, not just the optimized one. Even baseline was at 11W instead of 54W.
2. The old method was **most biased against Julia** (measured correctly at ~59W) vs. Python (under-estimated at ~11W).
3. With EMI, **all scripts are measured with the same hardware metric**, and the comparison is fair.
4. Julia's energy advantage was **masked by the measurement bias** — it went from "same as baseline" to "41.7% better than baseline."

### Why Joblib Multiprocessing Was Under-Reported

Technical explanation of the root cause:

1. CodeCarbon's estimation loop uses `psutil.cpu_percent(interval=1)` to sample CPU utilization once per second
2. `psutil.cpu_percent()` only reports utilization of the **current process** by default
3. Joblib (used by scikit-learn's `n_jobs=-1`) spawns **separate subprocesses** (the "Loky" backend)
4. These child processes are invisible to CodeCarbon's psutil polling because they are different PIDs
5. Result: reported CPU utilization for the Python process is near 0% (just waiting for children)
6. The estimated power: `95W (TDP) × ~0% ≈ 0W`, clamped to a minimum of ~11W (system idle draw)

Julia's `Threads.@threads` does not suffer from this because threads share the same PID, so `psutil.cpu_percent()` correctly aggregates their activity.

**Lesson learned:** When using CodeCarbon on Windows to compare different parallelization strategies, always ensure EMI (or another hardware-level method) is active. The estimation fallback is not reliable for cross-framework comparisons.

### External vs. Internal Tracking

This project demonstrates two CodeCarbon integration patterns:

| Pattern | Description | Files | When to Use |
| --- | --- | --- | --- |
| **Internal** | `EmissionsTracker` wraps the Python code directly. Tracker starts → Python computation → tracker stops. | `script1.py`, `script2.py` | Best when the computation is in Python |
| **External** | CodeCarbon runs as a separate orchestrator that launches the computation as a subprocess (Julia, C++, etc.) and monitors system-wide energy. | `script3/run_script3.py` | Necessary when the computation is in a non-Python language. Also useful for isolating measurement overhead. |

The `run_script3.py` orchestrator:

1. Deletes any previous emissions CSV (CodeCarbon appends by default)
2. Starts CodeCarbon tracking
3. Launches `julia -t auto --project=script3 script3/script3.jl` as a subprocess
4. Stops tracking when the subprocess exits
5. Saves the emissions report

This is the cleanest approach for measuring non-Python code — the Julia process has zero knowledge of Python or CodeCarbon.

---

## Results

### Final Comparison Table

| Script | Duration (s) | Energy (Wh) | CO₂ (g) | Avg MCC | Energy Reduction | Time Reduction |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline Python** | 73.22 s | 1.738 Wh | 0.575 g | 0.5990 | — | — |
| **Optimized Python** | 43.41 s | 0.993 Wh | 0.328 g | 0.5946 | **−42.88%** | **−40.71%** |
| **Julia Native** | 30.89 s | 0.920 Wh | 0.304 g | 0.5904 | **−47.07%** | **−57.81%** |

### Key Takeaways

| Conclusion | Detail |
| --- | --- |
| **Variant E delivers superior balance** | Combining 60 regularized trees, 60% feature selection, and single-threaded Python cuts energy by **42.9%** while preserving **99.3% of baseline MCC** (0.5946 vs 0.5990). |
| **Row subsampling avoided** | Empirical testing showed 50% row subsampling causes severe MCC degradation on imbalanced biomedical data (e.g. Sepsis MCC drops from 0.43 to 0.35). |
| **Julia is the most efficient** | Completes in **30.89 s** (−57.8% duration) and uses **0.920 Wh** (−47.1% energy) with native threading. |
| **Measurement fidelity** | With Windows EMI hardware counters, accurate RAPL energy readings are captured across Python and Julia without process-visibility blind spots. |
### Visual Comparison of CPU Power Readings

```text
CPU Power (Watts) — Higher is better (means the real consumption is captured)
┌────────────────────────────────────────────────────────────┐
│  Old (estimation)     New (EMI hardware counters)          │
│                                                             │
│  Script1  11W ░░░░     Script1  54W ████████████████████   │
│  Script2  12W ░░░░     Script2  61W █████████████████████▌ │
│  Script3  59W ███████  Script3  63W ██████████████████████▏│
└────────────────────────────────────────────────────────────┘
  Julia was always measured correctly (threads stay in-process).
  Python joblib was under-reported 5× because child PIDs were missed.
```

---

## Datasets

Five bio/health informatics datasets from peer-reviewed open-access publications:

| Dataset | File | Target | Rows | Features |
| --- | --- | --- | --- | --- |
| Neuroblastoma (YM2018) | `10_7717_peerj_5665_dataYM2018_neuroblastoma.csv` | Binary | ~500–1000 | ~6–12 |
| Pediatric Brain Tumor (Belgrade 2021) | `dataset_Belgrade2021_pediatric_brain_tumor_...` | Multiclass | ~300–600 | ~8–16 |
| Colorectal Cancer EHRs (Taipei 2018) | `dataset_Taipei2018_colorectal_cancer_EHRs_...` | Binary | ~1000–2000 | ~10–20 |
| Sepsis/SIRS | `journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv` | Binary | ~500–1000 | ~6–12 |
| Depression/Heart Failure | `journal.pone.0158570_S2File_depression_heart_failure.csv` | Binary | ~500–1000 | ~8–15 |

Each dataset has the target variable in the **last column**.

---

## Project Structure

```text
├── datasets/                                  # Bio/health informatics CSV datasets
│   ├── 10_7717_peerj_5665_dataYM2018_neuroblastoma.csv
│   ├── dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv
│   ├── dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv
│   ├── journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv
│   └── journal.pone.0158570_S2File_depression_heart_failure.csv
├── script1.py                                 # Baseline: RF (100 trees, 100 splits, no n_jobs)
├── script2.py                                 # Optimized: RF (60 trees, regularized, 60% features, n_jobs=1)
├── script3/
│   ├── Project.toml                           # Julia dependencies (DecisionTree, MLJ, etc.)
│   ├── script3.jl                             # Julia native script (Threads.@threads)
│   └── run_script3.py                         # External CodeCarbon orchestrator
├── compare_reports.py                         # Aggregate results → comparison.csv
├── requirements.txt                           # Python dependencies (see EMI note)
├── LICENSE                                    # MIT license
├── CodeCarbon reports/                        # Auto-generated: emissions CSV from CodeCarbon
├── mcc reports/                               # Auto-generated: MCC results per dataset
└── comparison reports/                        # Auto-generated: comparison.csv
```

---

## How to Reproduce

### Prerequisites

- **Windows 10/11** (required for EMI hardware measurement)
- **Python 3.9+**
- **Julia 1.9+** (only for script3)
- CPU with RAPL support (Intel Sandy Bridge+ or AMD Zen+)

### Installation: CodeCarbon from `windows_support`

This is the **critical step** for accurate measurements on Windows.

```bash
# ❌ NOT this (standard PyPI version lacks EMI):
# pip install codecarbon

# ✅ DO this (windows_support branch with EMI):
python -m venv .venv
.venv\Scripts\activate
pip install git+https://github.com/mlco2/codecarbon.git@windows_support
pip install pandas numpy scikit-learn
```

**Verification:** Run a quick test, then check the emissions CSV. The `cpu_power` column should show realistic values (~30–90 W depending on workload), and in the CodeCarbon console output you should see:

```text
CPU Tracking Method: Windows EMI
```

If you see `CPU Tracking Method: cpu_load` or very low power (5–15 W), EMI is not active and measurements will be unreliable.

### Installation: Julia Environment

```bash
julia --project=script3 -e 'using Pkg; Pkg.instantiate()'
```

If you encounter dependency issues:

```bash
julia --project=script3 -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "MLJ", "MLJDecisionTreeInterface", "StatsBase"])'
```

### Run All Experiments

```bash
# Activate venv (if not already), then:

python script1.py                # Baseline (≈1.2 min)
python script2.py                # Optimized (≈0.8 min)
python script3/run_script3.py    # Julia (≈0.6 min)
```

### Generate Comparison

```bash
python compare_reports.py
```

Outputs:

- `comparison reports/comparison.csv` — numeric summary table with percentage reductions

---

## Known Pitfalls & Lessons Learned

| Pitfall | Symptom | Solution |
| --- | --- | --- |
| **CodeCarbon underestimates power on Windows** | CPU power readings of ~5–15W | Install from `windows_support` branch |
| **Joblib multiprocessing invisible to psutil** | `cpu_utilization_percent` near 0% for Python scripts using `n_jobs=-1` | Use EMI for accurate hardware readings; or use `per_cpu=True` in psutil and sum across all cores |
| **CodeCarbon appends to existing CSV** | Second run shows double the expected duration | Delete old emissions CSV before each run (done automatically by `run_script3.py`) |
| **Julia path separators mismatch** | MCC report paths use `\` on Windows but comparison expects `/` | Normalize with `replace("\\" => "/")` (see `script3.jl:63`) |
| **Cold start vs. warm JIT** | Julia timing includes compilation | Accept as realistic for batch workloads; for server benchmarks, precompile |
| **Cross-platform EMI availability** | EMI only works on Windows 10/11 | On Linux, CodeCarbon reads RAPL from `/sys/class/powercap` — no special setup needed |
| **Random seed sensitivity** | Different runs may give slightly different energy values | Run multiple times and average; seeds are fixed within each experiment for reproducibility |

---

## License

This project is released under the [MIT License](LICENSE).
