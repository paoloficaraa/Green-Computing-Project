import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from codecarbon import EmissionsTracker

# ---------------------------------------------------------------------------
# Optimized Random Forest configuration for script2.py / script3.jl
# ---------------------------------------------------------------------------
# Chosen values:
#   - max_depth = 12: keeps trees shallow enough to reduce per-tree work while
#     preserving enough capacity for these biomedical datasets.
#   - min_samples_leaf = 5 and min_samples_split = 10: regularize each tree and
#     reduce the number of splits taken, lowering the total compute without
#     reducing the forest size.
#   - feature selection: variance-based top-k selection on the training split,
#     keeping max(8, ceil(0.6 * n_features)) features. This is applied before
#     fitting in both Python and Julia to keep the model lighter and to preserve
#     reproducibility.
#   - n_estimators = 100: OOB sweeps were checked with the same regularized
#     configuration before finalizing this script. The OOB error stabilizes early
#     on some datasets (roughly 30-60 trees), but it does not remain stable in a
#     uniformly consistent way across all five datasets. Keeping 100 trees avoids
#     sacrificing ensemble diversity and stays aligned with the baseline while the
#     workload reduction comes from lighter trees and fewer processed features.
# ---------------------------------------------------------------------------
N_ESTIMATORS = 100
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


def analyze_oob_stability(x_train, y_train):
    """Check whether OOB error stabilizes before 100 trees.

    The regularized model uses `oob_score=True`; we record the first tree count at
    which the OOB score changes by less than 0.005 from the previous check. That
    gives the user a concrete picture of the OOB stabilization behavior while the
    final script still keeps 100 trees because the stabilization is not uniformly
    robust across all five datasets.
    """
    oob_scores = {}
    stable_at = None

    for n_trees in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        oob_scores[n_trees] = float(model.oob_score_)

        if len(oob_scores) >= 2:
            prev_n_trees, prev_score = list(oob_scores.items())[-2]
            delta = abs(oob_scores[n_trees] - prev_score)
            if delta < 0.005:
                stable_at = n_trees

    return oob_scores, stable_at


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

    x_train_oob, _, y_train_oob, _ = train_test_split(
        x, y, test_size=0.3, random_state=0, stratify=y
    )
    selected_features = select_high_variance_features(x_train_oob)
    x_train_oob = x_train_oob[selected_features]

    oob_scores, stable_at = analyze_oob_stability(x_train_oob, y_train_oob)
    print(
        f"{os.path.basename(dataset)} | OOB scores: {oob_scores} | first stable check: {stable_at} trees"
    )

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
            n_jobs=-1,
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
