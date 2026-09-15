import os
from pathlib import Path

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS = [
    "datasets/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv",
    "datasets/dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv",
    "datasets/dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv",
    "datasets/journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv",
    "datasets/journal.pone.0158570_S2File_depression_heart_failure.csv",
]

VARIANTS = {
    "depth_only": {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "filter_features": False,
    },
    "filter_only": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "filter_features": True,
    },
    "trees_only": {
        "n_estimators": 60,
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "filter_features": False,
    },
}


def select_high_variance_features(x_train, ratio=0.6, min_keep=8):
    variances = x_train.var(axis=0, ddof=0).fillna(0.0)
    n_keep = min(max(min_keep, int(np.ceil(len(variances) * ratio))), len(variances))
    return variances.nlargest(n_keep).index.tolist()


def run_variant(name, settings):
    emissions_dir = PROJECT_ROOT / "CodeCarbon reports"
    mcc_dir = PROJECT_ROOT / "mcc reports"
    emissions_dir.mkdir(exist_ok=True)
    mcc_dir.mkdir(exist_ok=True)

    tracker = EmissionsTracker(
        output_file=str(emissions_dir / f"emissions_ablation_{name}.csv")
    )
    tracker.start()
    averages_mcc = {}
    try:
        for dataset in DATASETS:
            dataset_path = PROJECT_ROOT / dataset
            if not dataset_path.exists():
                print(f"Warning: {dataset} not found. Skipping.")
                continue

            df = pd.read_csv(dataset_path)
            # Drop post-outcome column: capped at 730-day follow-up for
            # survivors, so it leaks the target.
            df = df.drop(columns=["Time from HF to Death (days)"], errors="ignore")
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            scores = []

            for seed in range(100):
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.3, random_state=seed, stratify=y
                )
                if settings["filter_features"]:
                    selected = select_high_variance_features(x_train)
                    x_train = x_train[selected]
                    x_test = x_test[selected]

                model = RandomForestClassifier(
                    n_estimators=settings["n_estimators"],
                    max_depth=settings["max_depth"],
                    min_samples_leaf=settings["min_samples_leaf"],
                    min_samples_split=settings["min_samples_split"],
                    random_state=42,
                    n_jobs=1,
                )
                model.fit(x_train, y_train)
                scores.append(matthews_corrcoef(y_test, model.predict(x_test)))

            averages_mcc[dataset] = float(np.mean(scores))
    finally:
        tracker.stop()

    pd.DataFrame(
        [(dataset, score) for dataset, score in averages_mcc.items()],
        columns=["dataset", "mcc"],
    ).to_csv(mcc_dir / f"mcc_report_ablation_{name}.csv", index=False)


def main():
    for name, settings in VARIANTS.items():
        print(f"Running ablation: {name}")
        run_variant(name, settings)


if __name__ == "__main__":
    main()