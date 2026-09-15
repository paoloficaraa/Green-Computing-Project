from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean, stdev

EMISSIONS_DIR = Path("CodeCarbon reports")
MCC_DIR = Path("mcc reports")
OUT_DIR = Path("comparison reports")

SCRIPTS = [
    ("script1", "Baseline Python"),
    ("script2", "Optimized Python"),
    ("script3", "Julia"),
]


N_REPEATS = 3


def read_emissions_runs(script_id: str) -> dict | None:
    """Average emissions_script<id>_r1..rN.csv runs (fallback: single canonical file)."""
    run_paths = [EMISSIONS_DIR / f"emissions_{script_id}_r{k}.csv" for k in range(1, N_REPEATS + 1)]
    run_paths = [p for p in run_paths if p.exists()]
    if not run_paths:
        fallback = EMISSIONS_DIR / f"emissions_{script_id}.csv"
        run_paths = [fallback] if fallback.exists() else []
    if not run_paths:
        return None
    durations, energies, co2s = [], [], []
    for path in run_paths:
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[-1]
        durations.append(float(row["duration"]))
        energies.append(float(row["energy_consumed"]) * 1000.0)
        co2s.append(float(row["emissions"]) * 1000.0)
    if not durations:
        return None
    spread = lambda xs: stdev(xs) if len(xs) > 1 else 0.0
    return {
        "n_runs": len(durations),
        "duration_s": mean(durations),
        "duration_std_s": spread(durations),
        "energy_Wh": mean(energies),
        "energy_std_Wh": spread(energies),
        "co2_g": mean(co2s),
    }


def read_avg_mcc(path: Path) -> float | None:
    """Read an MCC report and return the mean MCC across datasets."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        values = [float(r["mcc"]) for r in reader]
    return mean(values) if values else None


def pct_reduction(base: float, value: float) -> float | None:
    if base == 0:
        return None
    return (base - value) / base * 100.0


def main():
    rows = []

    for script_id, label in SCRIPTS:
        mcc_path = MCC_DIR / f"mcc_report_{script_id}.csv"
        if not mcc_path.exists():
            print(f"Warning: {mcc_path} not found. Skipping {script_id}.")
            continue

        emissions = read_emissions_runs(script_id)
        if emissions is None:
            print(f"Warning: no emissions runs found. Skipping {script_id}.")
            continue
        print(f"{script_id}: averaging {emissions['n_runs']} run(s).")
        avg_mcc = read_avg_mcc(mcc_path)

        rows.append(
            {
                "script": script_id,
                "label": label,
                "duration_s": emissions["duration_s"],
                "duration_std_s": emissions["duration_std_s"],
                "energy_Wh": emissions["energy_Wh"],
                "energy_std_Wh": emissions["energy_std_Wh"],
                "co2_g": emissions["co2_g"],
                "avg_mcc": avg_mcc,
            }
        )

    if len(rows) < 1:
        print("No valid reports found. Nothing to compare.")
        return

    # Add percentage reductions vs script1 (baseline)
    baseline = rows[0]
    for row in rows:
        if row["script"] == baseline["script"]:
            row["energy_reduction_pct"] = None
            row["co2_reduction_pct"] = None
            row["time_reduction_pct"] = None
        else:
            row["energy_reduction_pct"] = pct_reduction(
                baseline["energy_Wh"], row["energy_Wh"]
            )
            row["co2_reduction_pct"] = pct_reduction(baseline["co2_g"], row["co2_g"])
            row["time_reduction_pct"] = pct_reduction(
                baseline["duration_s"], row["duration_s"]
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write CSV
    csv_path = OUT_DIR / "comparison.csv"
    fieldnames = [
        "script",
        "label",
        "duration_s",
        "duration_std_s",
        "energy_Wh",
        "energy_std_Wh",
        "co2_g",
        "avg_mcc",
        "energy_reduction_pct",
        "co2_reduction_pct",
        "time_reduction_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison written to:")
    print(f"  - {csv_path}")


if __name__ == "__main__":
    main()
