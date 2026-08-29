import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from codecarbon import EmissionsTracker

# ---------------------------------------------------------------------------
# Optimized Random Forest configuration for script2.py (Variant E)
# ---------------------------------------------------------------------------
# Chosen values:
#   - n_estimators = 60: achieves a direct 40% reduction in tree construction
#     workload while preserving strong ensemble diversity and variance reduction,
#     avoiding the predictive collapse on smaller datasets caused by aggressive
#     row subsampling.
#   - max_depth = 12: keeps trees shallow enough to prevent unnecessary deep
#     branching on larger datasets while retaining sufficient capacity for
#     biomedical classification.
#   - min_samples_leaf = 5 and min_samples_split = 10: regularize each tree and
#     prune fine-grained split evaluations, lowering node construction cost.
#   - feature selection: variance-based top-k selection on the training split,
#     keeping max(8, ceil(0.6 * n_features)) features. This is applied identically
#     in both Python and Julia to lighten computation and maintain reproducibility.
#   - single-threaded execution (n_jobs=1): eliminates joblib multiprocessing
#     IPC and thread synchronization overhead, which otherwise wastes CPU package
#     energy on tiny biomedical datasets (100-1000 samples).
#   - no row subsampling: preserves the full training distribution, protecting
#     clinical MCC performance across unbalanced biomedical datasets (e.g. Sepsis).
# ---------------------------------------------------------------------------
N_ESTIMATORS = 60
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 5
MIN_SAMPLES_SPLIT = 10
FEATURE_KEEP_RATIO = 0.6
FEATURE_MIN_KEEP = 8


def select_high_variance_features(x_train, min_keep=FEATURE_MIN_KEEP, ratio=FEATURE_KEEP_RATIO):
    """Keep the highest-variance columns in a deterministic, reproducible way."""
    variances = x_train.var(axis=0, ddof=0).fillna(0.0)
    n_features = len(variances)
    n_keep = min(max(min_keep, int(np.ceil(n_features * ratio))), n_features)
    selected = variances.nlargest(n_keep).index.tolist()
    return selected


datasets = [
    "datasets/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv",
    "datasets/dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv",
    "datasets/dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv",
    "datasets/journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv",
    "datasets/journal.pone.0158570_S2File_depression_heart_failure.csv",
]

tracker = EmissionsTracker(output_file="CodeCarbon reports/emissions_script2.csv")
tracker.start()

averages_mcc = {}

for dataset in datasets:
    if not os.path.exists(dataset):
        print(f"Warning: File {dataset} not found. Skip this dataset.")
        continue

    df = pd.read_csv(dataset)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    array_mcc = []
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=i, stratify=y
        )
        selected_features = select_high_variance_features(x_train)
        x_train = x_train[selected_features]
        x_test = x_test[selected_features]

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            random_state=42,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        array_mcc.append(matthews_corrcoef(y_test, y_pred))

    averages_mcc[dataset] = np.mean(array_mcc)

tracker.stop()

os.makedirs("mcc reports", exist_ok=True)
report_df = pd.DataFrame(
    [(name, score) for name, score in averages_mcc.items()],
    columns=["dataset", "mcc"],
)
report_df.to_csv("mcc reports/mcc_report_script2.csv", index=False)
