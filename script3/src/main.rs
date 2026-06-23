use std::path::Path;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use polars::prelude::*;
use ndarray::Array2;
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use rayon::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let datasets = vec![
        "datasets/10_7717_peerj_5665_dataYM2018_neuroblastoma.csv",
        "datasets/dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv",
        "datasets/dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv",
        "datasets/journal.pone.0148699_S1_Text_Sepsis_SIRS_EDITED.csv",
        "datasets/journal.pone.0158570_S2File_depression_heart_failure.csv"
    ];

    let mut csv_file = File::create("mcc reports/mcc_report_script3.csv")?;
    writeln!(csv_file, "Dataset,Average_MCC")?;

    for dataset in datasets {
        if !Path::new(dataset).exists() {
            println!("Attenzione: File {} non trovato. Salto.", dataset);
            continue;
        }

        let df = CsvReadOptions::default()
            .with_has_header(true)
            .with_infer_schema_length(None)
            .try_into_reader_with_file_path(Some(dataset.into()))?
            .finish()?;

        let target_col = df.columns().last().unwrap().name().to_string();

        let max_val = df
            .column(&target_col)
            .unwrap()
            .as_materialized_series()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .max()
            .unwrap_or(1.0);

        let max_class = max_val.round() as i32;
        let iterations = 100;

        let total_mcc: f64 = (0..iterations)
            .into_par_iter()
            .map(|i| {
                let df_shuffled = df
                    .sample_n_literal(df.height(), false, true, Some(i as u64))
                    .unwrap();
                
                let n = df_shuffled.height();
                let test_count = (n as f64 * 0.3).round() as usize; 
                let train_count = n - test_count;

                let train_df = df_shuffled.slice(0, train_count);
                let test_df = df_shuffled.slice(train_count as i64, test_count);

                let x_train_nd: Array2<f64> = train_df
                    .drop(target_col.as_str())
                    .unwrap()
                    .to_ndarray::<Float64Type>(IndexOrder::C)
                    .unwrap();

                let x_test_nd: Array2<f64> = test_df
                    .drop(target_col.as_str())
                    .unwrap()
                    .to_ndarray::<Float64Type>(IndexOrder::C)
                    .unwrap();

                let xn_tr = x_train_nd.nrows();
                let xc_tr = x_train_nd.ncols();
                let (x_train_data, _x_off) = x_train_nd.into_raw_vec_and_offset();
                let x_train_dense = DenseMatrix::new(xn_tr, xc_tr, x_train_data, false).unwrap();

                let xn_te = x_test_nd.nrows();
                let xc_te = x_test_nd.ncols();
                let (x_test_data, _x_off2) = x_test_nd.into_raw_vec_and_offset();
                let x_test_dense = DenseMatrix::new(xn_te, xc_te, x_test_data, false).unwrap();

                let y_train_vec: Vec<i32> = train_df
                    .column(&target_col)
                    .unwrap()
                    .as_materialized_series()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|opt_v| {
                        let val = opt_v.unwrap_or(0.0).round() as i32;
                        if val == max_class { 1 } else { 0 }
                    })
                    .collect();

                let y_test_vec: Vec<i32> = test_df
                    .column(&target_col)
                    .unwrap()
                    .as_materialized_series()
                    .cast(&DataType::Float64)
                    .unwrap()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|opt_v| {
                        let val = opt_v.unwrap_or(0.0).round() as i32;
                        if val == max_class { 1 } else { 0 }
                    })
                    .collect();

                let params = RandomForestClassifierParameters::default();
                let model = RandomForestClassifier::fit(&x_train_dense, &y_train_vec, params).unwrap();
                let y_pred = model.predict(&x_test_dense).unwrap();

                calculate_mcc(&y_test_vec, &y_pred)
            })
            .sum();

        let avg_mcc = total_mcc / (iterations as f64);
        
        let file_name = Path::new(dataset)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(dataset);

        writeln!(csv_file, "{},{:.4}", file_name, avg_mcc)?;
    }
    Ok(())
}

fn calculate_mcc(y_true: &[i32], y_pred: &[i32]) -> f64 {
    let mut tp: u64 = 0;
    let mut tn: u64 = 0;
    let mut fp: u64 = 0;
    let mut fn_val: u64 = 0;

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        let t_bool = t == 1;
        let p_bool = p == 1;

        if t_bool && p_bool { tp += 1; }
        else if !t_bool && !p_bool { tn += 1; }
        else if !t_bool && p_bool { fp += 1; }
        else if t_bool && !p_bool { fn_val += 1; }
    }

    let tp_f = tp as f64;
    let tn_f = tn as f64;
    let fp_f = fp as f64;
    let fn_f = fn_val as f64;

    let numerator = (tp_f * tn_f) - (fp_f * fn_f);
    let denominator = (tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f);
    let denom_sqrt = denominator.sqrt();

    if denom_sqrt < f64::EPSILON { 0.0 } else { numerator / denom_sqrt }
}