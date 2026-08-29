using CSV
using DataFrames
using Random
using StatsBase
using MLJ

# ---------------------------------------------------------------------------
# Julia implementation notes (Variant E)
# ---------------------------------------------------------------------------
# Chosen values:
#   - max_depth = 12, min_samples_leaf = 5, min_samples_split = 10.
#   - feature selection: variance-based top-k filtering on the training split,
#     keeping max(8, ceil(0.6 * n_features)) features, identical to Python.
#   - n_trees = 60: aligned with the optimized Python implementation (Variant E),
#     reducing tree building cost by 40% while preserving ensemble diversity,
#     variance reduction, and high MCC performance.
#   - parallel execution: native Julia multi-threading (Threads.@threads) over
#     the 100 hold-out splits in a single shared memory process.
# ---------------------------------------------------------------------------
const MODEL_N_TREES = 60
const MAX_DEPTH = 12
const MIN_SAMPLES_LEAF = 5
const MIN_SAMPLES_SPLIT = 10
const FEATURE_KEEP_RATIO = 0.6
const FEATURE_MIN_KEEP = 8

RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree verbosity=0

project_root = dirname(@__DIR__)
datasets = [
    joinpath(project_root, "datasets", "10_7717_peerj_5665_dataYM2018_neuroblastoma.csv"),
    joinpath(project_root, "datasets", "dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv"),
    joinpath(project_root, "datasets", "dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv"),
    joinpath(project_root, "datasets", "journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv"),
    joinpath(project_root, "datasets", "journal.pone.0158570_S2File_depression_heart_failure.csv")
]

function select_high_variance_features(X_train::DataFrame)
    variances = Float64[]
    for col in eachcol(X_train)
        push!(variances, var(skipmissing(Vector(col))))
    end

    n_features = length(variances)
    n_keep = min(max(FEATURE_MIN_KEEP, Int(ceil(n_features * FEATURE_KEEP_RATIO))), n_features)
    order = sortperm(variances, rev=true)
    selected_cols = names(X_train)[order[1:n_keep]]
    return selected_cols
end

averages_mcc = Dict{String,Float64}()

for dataset in datasets
    if !isfile(dataset)
        println("Warning: File $dataset not found. Skip this dataset.")
        continue
    end

    df = CSV.read(dataset, DataFrame)
    X = df[:, 1:(end-1)]
    y_raw = df[:, end]
    y = coerce(y_raw, OrderedFactor)

    ThreadLocal = Array{Float64}(undef, 100)

    Threads.@threads for i in 0:99
        train_idx, test_idx = partition(eachindex(y), 0.7, stratify=y, rng=i)

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        selected_cols = select_high_variance_features(X_train)
        X_train = X_train[:, selected_cols]
        X_test = X_test[:, selected_cols]

        model = RandomForestClassifier(
            n_trees=MODEL_N_TREES,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            min_samples_split=MIN_SAMPLES_SPLIT,
            rng=i,
        )

        mach = machine(model, X_train, y_train)
        fit!(mach, verbosity=0)

        y_pred = predict_mode(mach, X_test)
        ThreadLocal[i+1] = MatthewsCorrelation()(y_pred, y_test)
    end

    rel_path = replace(relpath(dataset, project_root), "\\" => "/")
    averages_mcc[rel_path] = mean(ThreadLocal)
end

mkpath(joinpath(project_root, "mcc reports"))
report_df = DataFrame(
    dataset=collect(keys(averages_mcc)),
    mcc=collect(values(averages_mcc))
)
sort!(report_df, :dataset)
CSV.write(joinpath(project_root, "mcc reports", "mcc_report_script3.csv"), report_df)
