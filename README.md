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
  - [The Fix: EMI Backend](#the-fix-emi-backend-merged-upstream)
  - [Before vs. After EMI Comparison](#before-vs-after-emi-comparison)
  - [Why Joblib Multiprocessing Was Under-Reported](#why-joblib-multiprocessing-was-under-reported)
  - [External vs. Internal Tracking](#external-vs-internal-tracking)
- [Results](#results)
  - [Final Comparison Table](#final-comparison-table)
  - [Ablation Study](#ablation-study-one-factor-hardware-matched)
  - [Visual Comparison of CPU Power Readings](#visual-comparison-of-cpu-power-readings)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
  - [Prerequisites](#prerequisites)
  - [Installation: CodeCarbon with EMI backend](#installation-codecarbon-with-emi-backend)
  - [Installation: Julia Environment](#installation-julia-environment)
  - [Run All Experiments](#run-all-experiments)
  - [Generate Comparison](#generate-comparison)
- [Known Pitfalls & Lessons Learned](#known-pitfalls--lessons-learned)
- [License](#license)

---

## Project Overview

Comparative analysis of energy consumption for Random Forest classification across **Python (scikit-learn)** and **Julia (native DecisionTree.jl)** implementations, measured via hardware-level CPU energy counters on Windows 11. Random Forest is the deliberate model family: CPU-native and competitive on small tabular EHR data with no GPU in the loop; the savings below come from configuration. The core research question:

> **How much energy can be saved while preserving model quality by targeting the computational cost inside each tree and forest sizing, rather than by relying on aggressive row subsampling?**

The final strategy combines 60 regularized trees (`n_estimators=60`, `max_depth=12`, `min_samples_leaf=5`, `min_samples_split=10`), variance-based feature selection (top 60%, floor 8), and single-threaded Python execution (thread-pool overhead exceeds any gain on small tabular datasets). This achieves a **33.6% energy reduction in Python** and **45.5% in Julia** (mean of 3 runs each) while preserving an average MCC of **0.4539 / 0.4574** (against 0.4688 baseline).

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
- variance-based feature selection before training, keeping the top 60% of features by variance, minimum 8
- single-threaded execution (`n_jobs=1`) because thread-pool overhead exceeds any gain on small EHR tables (the Cython splitter releases the GIL, so threads work but cost more than they save here)
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

> **Threading fairness note:** Python (`script2.py`) runs single-threaded (`n_jobs=1`) because threading overhead exceeds any gain on these small tables, while Julia (`script3.jl`) uses `Threads.@threads` shared-memory parallelism across splits. The Julia-vs-Python gap mixes language/runtime efficiency with threading strategy.
> **Leakage note:** all pipelines drop `Time from HF to Death (days)` in the depression/HF cohort. Survivors are capped at the 730-day follow-up (cohort mean 730.0), which identifies them. Keeping the column scored MCC 0.9985.

## Measurement Methodology (Key Learning)

This section covers the measurement problem behind every number below: CodeCarbon's default Windows method estimates instead of measuring.

### How CodeCarbon Measures Energy

CodeCarbon computes energy consumption as:

```text
Energy = CPU_Power × Duration + GPU_Power × Duration + RAM_Power × Duration
```

CodeCarbon samples CPU power every second and averages the samples. Without a hardware reading, it estimates:

```text
CPU_Power_estimated = TDP × (cpu_utilization_percent / 100)
```

Where:

- **TDP** (Thermal Design Power) is a constant, the chip's maximum rated thermal output
- **cpu_utilization_percent** is polled from `psutil.cpu_percent()` every second

### The Three Tracking Modes

CodeCarbon has three CPU tracking methods, in order of preference:

| # | Method | Source | Accuracy | Platforms |
| --- | --- | --- | --- | --- |
| 1 | **RAPL** (Running Average Power Limit) | Hardware energy counters via `perf_event_open()` (Linux) or `IntelPowerGadget` (macOS) | **Very High** (real hardware registers) | Linux, macOS |
| 2 | **Windows EMI** (Energy Meter Interface) | Hardware energy counters via Windows Power Management API | **Very High** (same RAPL registers, different OS API) | Windows only |
| 3 | **CPU Load Estimation** | `TDP × cpu_utilization_percent` | **Low** (estimated, not measured) | Fallback on all OS |

### The EMI Bug: What I Discovered

**The problem (fixed upstream):** CodeCarbon ≤3.2.8 on PyPI had Windows EMI support **disabled by default** in its tracking mode selection logic. On Windows it fell through to the estimation method (see [CodeCarbon PR #1263](https://github.com/mlco2/codecarbon/pull/1263)).

**Why it existed:** the EMI integration arrived as a pull request after those releases. EMI is merged upstream now: CodeCarbon 3.3.0 on PyPI measures via EMI out of the box, and every number below comes from 3.3.0.

**The result (at the time):** with CodeCarbon ≤3.2.8, every measurement on Windows used `TDP × cpu_load` estimation, which is unreliable.

### Diagnosing the Problem

I discovered the issue during result analysis. The initial measurements showed:

```text
# OLD (before EMI fix) - suspicious results
Script          CPU_Power   CPU_Util    Duration   Energy
Baseline        10.97W      ~8%         72.0s      0.960 Wh
Optimized       11.70W      ~0%         46.1s      0.648 Wh
Julia           58.93W      ~44%        35.7s      0.959 Wh
```

All three scripts run on the same i7-9700K, so their CPU power should match. It did not: Python showed ~11W, Julia ~59W.

| Script | Parallel Model | Process Visibility to psutil | Reported CPU Util | Estimated Power |
| --- | --- | --- | --- | --- |
| Baseline (no n_jobs) | None (single-threaded) | Main thread only | ~8% | ~11W (TDP=95W × 8%) |
| Optimized (n_jobs=-1) | joblib multiprocessing | Child processes not polled | ~0% | ~11W (effectively idle power) |
| Julia (Threads.@threads) | OS threads in same process | Same process, all threads | ~44% | ~59W |

**Julia measured correctly** because its threads live in the same process as the main Julia process, so `psutil.cpu_percent()` captured all of them. But scikit-learn's `n_jobs=-1` spawns separate child processes via `joblib.Parallel`, and the sampling loop never aggregated their CPU utilization.

**The comparison was unfair:** Optimized Python paid idle power (~11W, the system's background power) while Julia paid the full computing power (~59W).

### The Fix: EMI Backend (Merged Upstream)

Early runs used CodeCarbon from the `windows_support` branch (PR #1263). That branch no longer exists upstream. EMI was merged, and CodeCarbon 3.3.0 on PyPI measures via EMI out of the box:
```bash
pip install -r requirements.txt   # pins codecarbon==3.3.0
```
All headline numbers below come from 3.3.0 with the EMI backend confirmed in the tracker log.

**Verification:** After installation, CodeCarbon reports:

```text
CPU Tracking Method: Windows EMI
```

It reads the same hardware registers Linux exposes via `/sys/class/powercap`.

| Metric | Without EMI (≤3.2.8, estimation) | With EMI (3.3.0, mean of 3 runs) | Δ |
| --- | --- | --- | --- |
| **CPU Tracking Method** | `cpu_load` (estimation) | `Windows EMI` (hardware) | - |
| **Baseline CPU Power** | 10.97 W | 36.5 W | hardware reading |
| **Optimized CPU Power** | 11.70 W | 35.1 W | hardware reading |
| **Julia CPU Power** | 58.93 W | 37.6 W | hardware reading |

| Script | Energy (old method) | Energy (EMI, mean of 3) | Old Conclusion | Correct Conclusion |
| --- | --- | --- | --- | --- |
| Baseline | 0.960 Wh | 1.845 Wh | - (baseline) | - (baseline) |
| Optimized | 0.648 Wh | 1.226 Wh | −32.5% energy (seems good) | **−33.6%** energy |
| Julia | 0.959 Wh | 1.005 Wh | −0.1% energy (same as baseline!) | **−45.5%** energy |

**Key insights:**
1. The old method **under-estimated all Python scripts**, not just the optimized one.
2. The old method was **most biased against Julia** (a true ~59W reading) vs. Python (under-estimated at ~11W).
3. With EMI, **all scripts are measured with the same hardware metric**, and the comparison is fair.
4. Julia's energy advantage was **masked by the measurement bias** - it went from "same as baseline" to "45.5% better than baseline."

### Why Joblib Multiprocessing Was Under-Reported

Technical explanation of the root cause:

1. CodeCarbon's estimation loop uses `psutil.cpu_percent(interval=1)` to sample CPU utilization once per second
2. `psutil.cpu_percent()` only reports utilization of the **current process** by default
3. Joblib (used by scikit-learn's `n_jobs=-1`) spawns **separate subprocesses** (the "Loky" backend)
4. These child processes are invisible to CodeCarbon's psutil polling because they are different PIDs
5. Result: reported CPU utilization for the Python process is near 0% (just waiting for children)
6. The estimated power: `95W (TDP) × ~0% ≈ 0W`, clamped to a minimum of ~11W (system idle draw)

Julia's `Threads.@threads` avoids this: threads share one PID, so `psutil.cpu_percent()` aggregates all of them.

**Lesson learned:** meter Windows runs with EMI or another hardware method; the estimation fallback cannot support cross-framework comparisons.

This project uses two CodeCarbon integration patterns:
| --- | --- | --- | --- |
| **Internal** | `EmissionsTracker` wraps the Python code. Tracker starts → Python computation → tracker stops. | `script1.py`, `script2.py` | Best when the computation is in Python |
| **External** | CodeCarbon runs as a separate orchestrator that launches the computation as a subprocess (Julia, C++, etc.) and monitors system-wide energy. | `script3/run_script3.py` | Necessary when the computation is in a non-Python language. Also useful for isolating measurement overhead. |

The `run_script3.py` orchestrator:

1. Deletes any previous emissions CSV (CodeCarbon appends by default)
2. Starts CodeCarbon tracking
3. Launches `julia -t auto --project=script3 script3/script3.jl` as a subprocess
4. Stops tracking when the subprocess exits
5. Saves the emissions report

External tracking keeps the measured code clean: the Julia process needs no knowledge of Python or CodeCarbon.

## Results

### Final Comparison Table

| Script | Duration (s) | Energy (Wh) | CO₂ (g) | Avg MCC | Energy Reduction | Time Reduction |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline Python** | 72.45 ± 2.17 s | 1.845 ± 0.092 Wh | 0.610 g | 0.4688 | - | - |
| **Optimized Python** | 45.69 ± 0.79 s | 1.226 ± 0.092 Wh | 0.405 g | 0.4539 | **−33.6%** | **−36.9%** |
| **Julia Native** | 33.31 ± 1.39 s | 1.005 ± 0.031 Wh | 0.332 g | 0.4574 | **−45.5%** | **−54.0%** |

### Key Takeaways

| Conclusion | Detail |
| --- | --- |
| **Variant E is the balanced choice:** combining 60 regularized trees, 60% feature selection, and single-threaded Python cuts energy by **33.6%** while preserving **96.8% of baseline MCC** (0.4539 vs 0.4688). |
| **Row subsampling avoided** | We tested 50% row subsampling: sepsis MCC fell from 0.43 to 0.35 on imbalanced biomedical data. |
| **Julia is the fastest** | Runs in **33.31 s** (−54.0% duration) and uses **1.005 Wh** (−45.5% energy) with native threading. |
| **Measurement fidelity** | Windows EMI hardware counters meter Python and Julia from the same RAPL registers, with no process-visibility blind spots. |

### Ablation Study (One-Factor, Hardware-Matched)

To isolate each lever, I changed one setting at a time against baseline (same 5 datasets × 100 splits, single-threaded Python, same i7-9700K + EMI protocol as the headline benchmark):

| Variant | Changed lever | Energy (Wh) | Δ vs baseline | Mean MCC |
| --- | --- | --- | --- | --- |
| Depth-only | `max_depth=12` | 1.788 Wh | −3.1% (neutral) | 0.4655 |
| Filter-only | top-60% variance features | 1.854 Wh | +0.5% (neutral) | 0.4364 |
| Trees-only | `n_estimators=60` | 1.113 Wh | **−39.7%** | 0.4625 |

**Reading:** only ensemble sizing is an energy lever (−39.7% alone covers the full −33.6% combined saving). Depth capping alone is energy-neutral; variance filtering alone costs quality (sepsis MCC drops to 0.418 - rare acute signals get discarded) at neutral energy. The MCC recovery (0.4364 → 0.4539) comes from combining all three levers with the tightened leaf/split minima.

**Tuning-energy payback:** the three ablation runs cost 1.788 + 1.854 + 1.113 = 4.755 Wh one-time; the deployed config saves 0.619 Wh per 500-fit campaign, so the tuning cost needs ≈ 7.7 campaigns to pay back, within the first night of network-wide nightly retraining. Raw data: `CodeCarbon reports/emissions_ablation_*.csv`, `mcc reports/mcc_report_ablation_*.csv`.

### Visual Comparison of CPU Power Readings

```text
CPU Power (Watts) - Higher is better (means the real consumption is captured)
┌────────────────────────────────────────────────────────────┐
│  Old (estimation)     New (EMI hardware counters)          │
│                                                             │
│  Script1  11W ░░░░     Script1  37W ██████████████▌      │
│  Script2  12W ░░░░     Script2  35W █████████████▊       │
│  Script3  59W ███████  Script3  38W ███████████████      │
└────────────────────────────────────────────────────────────┘
  Julia was always measured correctly (threads stay in-process).
  Python joblib was under-reported 5× because child PIDs were missed.
```

---

## Datasets

Five bio/health informatics datasets from peer-reviewed open-access publications:

| Dataset | File | Target | Rows | Features |
| --- | --- | --- | --- | --- |
| Neuroblastoma (Ma et al. 2018) | `10_7717_peerj_5665_dataYM2018_neuroblastoma.csv` | Binary (outcome) | 169 | 12 |
| Pediatric Brain Tumor (Stanić et al. 2021) | `dataset_Belgrade2021_pediatric_brain_tumor_...` | Binary (survival) | 173 | 30 |
| Colorectal Cancer (Tai et al. 2018) | `dataset_Taipei2018_colorectal_cancer_EHRs_...` | Binary (mortality) | 999 | 31 |
| Sepsis/SIRS ICU (Güçyetmez & Atalan 2016) | `journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv` | Binary (ICU mortality) | 1257 | 15 |
| Depression/Heart Failure (Jani et al. 2016) | `journal.pone.0158570_S2File_depression_heart_failure.csv` | Binary (death) | 425 | 14* |

Each dataset has the target variable in the **last column**. \*14 features after dropping `Time from HF to Death (days)`: capped at the 730-day follow-up for survivors, it leaks the target (kept: MCC 0.9985).

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
├── run_ablation.py                            # One-factor ablation (depth/filter/trees variants)
├── compare_reports.py                         # Aggregate results → comparison.csv
├── generate_plots.py                          # Publication figures (duration/energy, MCC, Pareto)
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

### Installation: CodeCarbon with EMI Backend

This is the **critical step** for accurate measurements on Windows: a plain `pip install codecarbon` may resolve to a build whose Windows backend is the TDP × load estimator. This project pins the known-good build:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt   # pins codecarbon==3.3.0 (EMI merged upstream)
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
python run_ablation.py           # One-factor ablation, 3 variants (≈3.5 min)
```

Each pipeline was run 3× (`emissions_*_r1..r3.csv`); `compare_reports.py` averages the repeats into `comparison.csv` (mean ± SD over the 3 runs).

### Generate Comparison

```bash
python compare_reports.py        # headline benchmark table
python generate_plots.py         # duration/energy, MCC, Pareto figures
```

Outputs:

- `comparison reports/comparison.csv` - numeric summary table with percentage reductions
- `duration_energy_comparison.png`, `mcc_comparison.png`, `pareto_frontier.png` - report figures
- `CodeCarbon reports/emissions_ablation_*.csv` + `mcc reports/mcc_report_ablation_*.csv` - raw ablation data

---

## Known Pitfalls & Lessons Learned

| Pitfall | Symptom | Solution |
| --- | --- | --- |
| **CodeCarbon underestimates power on Windows** | CPU power readings of ~5–15W | Use CodeCarbon ≥3.3.0 with the EMI backend (pinned in `requirements.txt`) |
| **Joblib multiprocessing invisible to psutil** | `cpu_utilization_percent` near 0% for Python scripts using `n_jobs=-1` | Use EMI for accurate hardware readings; or use `per_cpu=True` in psutil and sum across all cores |
| **CodeCarbon appends to existing CSV** | Second run shows double the expected duration | Delete old emissions CSV before each run (done automatically by `run_script3.py`) |
| **Julia path separators mismatch** | MCC report paths use `\` on Windows but comparison expects `/` | Normalize with `replace("\\" => "/")` (see `script3.jl:63`) |
| **Cold start vs. warm JIT** | Julia timing includes compilation | Accept as realistic for batch workloads; for server benchmarks, precompile |
| **Cross-platform EMI availability** | EMI only works on Windows 10/11 | On Linux, CodeCarbon reads RAPL from `/sys/class/powercap` - no special setup needed |
| **Random seed sensitivity** | Different runs may give slightly different energy values | Each pipeline runs 3× and `compare_reports.py` reports mean ± SD; model seeds stay fixed for reproducibility |

---

## License

This project is released under the [MIT License](LICENSE).
