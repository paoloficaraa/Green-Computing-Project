from __future__ import annotations

import csv
import os
from pathlib import Path
from statistics import mean

EMISSIONS_DIR = Path("CodeCarbon reports")
MCC_DIR = Path("mcc reports")
OUT_DIR = Path("comparison reports")

SCRIPTS = [
    ("script1", "Baseline Python"),
    ("script2", "Optimized Python"),
    ("script3", "Julia"),
]


def read_emissions(path: Path) -> dict | None:
    """Read the CodeCarbon CSV and return duration, energy (Wh), CO2 (g)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        if row is None:
            return None
    return {
        "duration_s": float(row["duration"]),
        "energy_Wh": float(row["energy_consumed"]) * 1000.0,
        "co2_g": float(row["emissions"]) * 1000.0,
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
        emissions_path = EMISSIONS_DIR / f"emissions_{script_id}.csv"
        mcc_path = MCC_DIR / f"mcc_report_{script_id}.csv"

        if not emissions_path.exists():
            print(f"Warning: {emissions_path} not found. Skipping {script_id}.")
            continue
        if not mcc_path.exists():
            print(f"Warning: {mcc_path} not found. Skipping {script_id}.")
            continue

        emissions = read_emissions(emissions_path)
        avg_mcc = read_avg_mcc(mcc_path)

        rows.append(
            {
                "script": script_id,
                "label": label,
                "duration_s": emissions["duration_s"],
                "energy_Wh": emissions["energy_Wh"],
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
        "energy_Wh",
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
