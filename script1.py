import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from codecarbon import EmissionsTracker

datasets = [
    "datasets/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv",
    "datasets/dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv",
    "datasets/dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv",
    "datasets/journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv",
    "datasets/journal.pone.0158570_S2File_depression_heart_failure.csv"
]

tracker = EmissionsTracker(output_file="CodeCarbon reports/emissions_script1.csv")
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
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        array_mcc.append(matthews_corrcoef(y_test, y_pred))
        
    averages_mcc[dataset] = np.mean(array_mcc)

tracker.stop()

os.makedirs("mcc reports", exist_ok=True)
report_df = pd.DataFrame(
    [(name, score) for name, score in averages_mcc.items()],
    columns=["dataset", "mcc"]
)
report_df.to_csv("mcc reports/mcc_report_script1.csv", index=False)