ENV["JULIA_PYTHONCALL_EXE"] = joinpath(@__DIR__, ".venv", "Scripts", "python.exe")

using CSV
using DataFrames
using Random
using StatsBase
using PythonCall
using MLJ

RandomForestClassifier = @load RandomForestClassifier pkg=MLJScikitLearnInterface verbosity=0

datasets = [
    "datasets/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv",
    "datasets/dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv",
    "datasets/dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv",
    "datasets/journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv",
    "datasets/journal.pone.0158570_S2File_depression_heart_failure.csv"
]

codecarbon = pyimport("codecarbon")
tracker = codecarbon.EmissionsTracker(output_file="CodeCarbon reports/emissions_script3.csv")
tracker.start()

averages_mcc = Dict{String,Float64}()

for dataset in datasets
    if !isfile(dataset)
        println("Warning: File $dataset not found. Skip this dataset.")
        continue
    end

    df = CSV.read(dataset, DataFrame)

    rng_sample = MersenneTwister(42)
    n_rows = nrow(df)
    sampled_indices = sample(rng_sample, 1:n_rows, div(n_rows, 2), replace=false)
    df_sampled = df[sampled_indices, :]

    X = df_sampled[:, 1:(end-1)]
    y_raw = df_sampled[:, end]

    y = coerce(y_raw, Multiclass)

    array_mcc = Float64[]

    for i in 0:99
        train_idx, test_idx = partition(eachindex(y), 0.7, stratify=y, rng=i)

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

        mach = machine(model, X_train, y_train)

        fit!(mach, verbosity=0)

        y_pred = predict_mode(mach, X_test)

        push!(array_mcc, MatthewsCorrelation()(y_pred, y_test))
    end

    averages_mcc[dataset] = mean(array_mcc)
end

tracker.stop()

mkpath("mcc reports")
report_df = DataFrame(
    dataset=collect(keys(averages_mcc)),
    mcc=collect(values(averages_mcc))
)
CSV.write("mcc reports/mcc_report_script3.csv", report_df)