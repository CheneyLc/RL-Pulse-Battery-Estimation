"""Figure 6: propagation of material and SOC errors to SOH estimation."""
import numpy as np
import pandas as pd

from ..common import FEATURES, material_files, predict_soh_continuous


def analyze_specific_impacts(df_test, M):
    clf_mat, le_mat, scaler = M["clf_mat"], M["le_mat"], M["scaler"]
    soc_models, soh_models = M["soc_models"], M["soh_models"]
    analysis_data = []

    for _, row in df_test.iterrows():
        u_raw = row[FEATURES].values.reshape(1, -1)
        u_s = scaler.transform(u_raw)
        true_mat = row["Mat"]
        true_soc = row["SOC"]
        true_soh = row["SOH"]

        wrong_mats = [m for m in material_files.keys() if m != true_mat]
        fake_mat = wrong_mats[0]

        soc_on_wrong_mat = soc_models[fake_mat]["model"].predict(u_s)[0]
        x_soh_wrong = np.hstack([u_raw, [[soc_on_wrong_mat]]])
        x_soh_wrong_s = soh_models[fake_mat]["scaler"].transform(x_soh_wrong)
        soh_on_wrong_mat = soh_models[fake_mat]["model"].predict(x_soh_wrong_s)[0]

        biased_soc = true_soc * 1.3
        x_soh_soc_err = np.hstack([u_raw, [[biased_soc]]])
        x_soh_soc_err_s = soh_models[true_mat]["scaler"].transform(x_soh_soc_err)
        soh_with_soc_bias = soh_models[true_mat]["model"].predict(x_soh_soc_err_s)[0]

        analysis_data.append({
            "True_Mat": true_mat,
            "SOC_Error_on_Wrong_Mat": abs(soc_on_wrong_mat - true_soc),
            "SOH_Error_on_Wrong_Mat": abs(soh_on_wrong_mat - true_soh),
            "SOH_Error_with_SOC_Bias": abs(soh_with_soc_bias - true_soh),
            "Normal_SOH_Error": abs(predict_soh_continuous(row, M, mode="hard") - true_soh)
        })

    return pd.DataFrame(analysis_data)


def soc_error_records(df_test, M):
    scaler, soc_models = M["scaler"], M["soc_models"]
    soc_error_records = []

    for _, row in df_test.iterrows():

        U = row[FEATURES].values.reshape(1, -1)
        U_scaled = scaler.transform(U)

        true_mat = row["Mat"]
        soc_true = row["SOC"]

        for mat_wrong in soc_models:
            if mat_wrong == true_mat:
                continue

            soc_cf = soc_models[mat_wrong]["model"].predict(U_scaled)[0]

            soc_error_records.append({
                "true_mat": true_mat,
                "assumed_mat": mat_wrong,
                "SOC_true": soc_true,
                "SOC_cf": soc_cf,
                "SOC_error": soc_cf - soc_true
            })

    return pd.DataFrame(soc_error_records)


def soh_error_records(df_test, M):
    scaler, soc_models, soh_models = M["scaler"], M["soc_models"], M["soh_models"]
    soh_error_records = []

    for _, row in df_test.iterrows():

        U = row[FEATURES].values.reshape(1, -1)
        true_mat = row["Mat"]

        soc_true = row["SOC"]
        soh_true = row["SOH"]

        for mat_wrong in soc_models:
            if mat_wrong == true_mat:
                continue

            soc_cf = soc_models[mat_wrong]["model"].predict(
                scaler.transform(U)
            )[0]

            X_soh = np.hstack([U, [[soc_cf]]])
            X_soh_scaled = soh_models[mat_wrong]["scaler"].transform(X_soh)
            soh_cf = soh_models[mat_wrong]["model"].predict(X_soh_scaled)[0]

            soh_error_records.append({
                "true_mat": true_mat,
                "assumed_mat": mat_wrong,
                "SOC_error": soc_cf - soc_true,
                "SOH_error": soh_cf - soh_true
            })

    return pd.DataFrame(soh_error_records)


def gain_records(df_test, M):
    soh_models = M["soh_models"]
    soc_deltas = np.linspace(-0.05, 0.05, 11)

    gain_records = []

    for _, row in df_test.iterrows():

        U = row[FEATURES].values.reshape(1, -1)
        mat = row["Mat"]

        soc_true = row["SOC"]
        soh_true = row["SOH"]

        for delta in soc_deltas:
            soc_perturbed = soc_true + delta

            X_soh = np.hstack([U, [[soc_perturbed]]])
            X_soh_scaled = soh_models[mat]["scaler"].transform(X_soh)
            soh_pred = soh_models[mat]["model"].predict(X_soh_scaled)[0]

            gain_records.append({
                "Mat": mat,
                "SOC_delta": delta,
                "SOH_delta": soh_pred - soh_true
            })

    return pd.DataFrame(gain_records)


def summary_table(df_soh_err):
    summary = []

    for key, g in df_soh_err.groupby(["true_mat", "assumed_mat"]):
        summary.append({
            "Case": f"{key[0]}→{key[1]}",
            "Mean_SOC_Error": g["SOC_error"].mean(),
            "Std_SOC_Error": g["SOC_error"].std(),
            "Mean_SOH_Error": g["SOH_error"].mean(),
            "Std_SOH_Error": g["SOH_error"].std()
        })

    return pd.DataFrame(summary)


def process_data(df_test, M):
    res_impact = analyze_specific_impacts(df_test, M)
    df_soc_err = soc_error_records(df_test, M)
    df_soh_err = soh_error_records(df_test, M)
    df_gain = gain_records(df_test, M)
    df_summary = summary_table(df_soh_err)
    print(df_summary)
    return {"res_impact.csv": res_impact, "df_soc_err.csv": df_soc_err,
            "df_soh_err.csv": df_soh_err, "df_gain.csv": df_gain,
            "df_summary.csv": df_summary}
