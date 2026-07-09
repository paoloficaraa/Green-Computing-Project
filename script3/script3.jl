using CSV
using DataFrames
using Random
using StatsBase
using MLJ

# Carica il modello nativo Julia DecisionTree (senza scikit-learn Python)
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree verbosity=0

project_root = dirname(@__DIR__)
datasets = [
    joinpath(project_root, "datasets", "10_7717_peerj_5665_dataYM2018_neuroblastoma.csv"),
    joinpath(project_root, "datasets", "dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv"),
    joinpath(project_root, "datasets", "dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv"),
    joinpath(project_root, "datasets", "journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv"),
    joinpath(project_root, "datasets", "journal.pone.0158570_S2File_depression_heart_failure.csv")
]

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

    # Vettore pre-allocato per raccogliere l'MCC in modo thread-safe
    array_mcc = Vector{Float64}(undef, 100)

    # Eseguiamo i 100 split in parallelo sfruttando i thread di Julia
    Threads.@threads for i in 0:99
        train_idx, test_idx = partition(eachindex(y), 0.7, stratify=y, rng=i)

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        # RF nativo Julia con 50 alberi
        # Usiamo un seed diverso per ogni iterazione per evitare race condition sul RNG globale
        model = RandomForestClassifier(n_trees=50, rng=i)

        mach = machine(model, X_train, y_train)

        fit!(mach, verbosity=0)

        y_pred = predict_mode(mach, X_test)

        array_mcc[i+1] = MatthewsCorrelation()(y_pred, y_test)
    end

    # Memorizziamo il percorso relativo esatto (es: "datasets/...") per essere compatibili con compare_reports.py
    rel_path = replace(relpath(dataset, project_root), "\\" => "/")
    averages_mcc[rel_path] = mean(array_mcc)
end

# Scrittura del report MCC
mkpath(joinpath(project_root, "mcc reports"))
report_df = DataFrame(
    dataset=collect(keys(averages_mcc)),
    mcc=collect(values(averages_mcc))
)
sort!(report_df, :dataset)
CSV.write(joinpath(project_root, "mcc reports", "mcc_report_script3.csv"), report_df)
