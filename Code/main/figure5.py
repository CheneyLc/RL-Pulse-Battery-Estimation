"""Figure 5: SOH estimation with (SOC-aware) and without (SOC-unaware) SOC input."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ..common import FEATURES, predict_soh_continuous


def train_soh_models_no_soc(df_train_used):
    soh_models_no_soc = {}

    for mat in df_train_used["Mat"].unique():

        idx = df_train_used["Mat"] == mat

        X_m = df_train_used.loc[idx, FEATURES].values
        y_m = df_train_used.loc[idx, "SOH"].values

        scaler_m = StandardScaler()
        X_m_scaled = scaler_m.fit_transform(X_m)

        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
        model.fit(X_m_scaled, y_m)

        soh_models_no_soc[mat] = {
            "model": model,
            "scaler": scaler_m
        }

        print(f"SOC-unaware SOH model trained for {mat}, samples={len(y_m)}")
    return soh_models_no_soc


def compare_metrics_by_material(df_test, y_no_soc, y_soc):

    records = []

    for mat in df_test["Mat"].unique():
        idx = df_test["Mat"] == mat

        y_t = df_test.loc[idx, "SOH"].values
        y_n = np.array(y_no_soc)[idx]
        y_s = np.array(y_soc)[idx]

        mask_n = ~np.isnan(y_n)
        mask_s = ~np.isnan(y_s)

        if mask_n.sum() < 20 or mask_s.sum() < 20:
            continue

        records.append({
            "Mat": mat,
            "MAE_noSOC": mean_absolute_error(y_t[mask_n], y_n[mask_n]),
            "MAE_SOC": mean_absolute_error(y_t[mask_s], y_s[mask_s]),
            "R2_noSOC": r2_score(y_t[mask_n], y_n[mask_n]),
            "R2_SOC": r2_score(y_t[mask_s], y_s[mask_s]),
        })

    df_cmp = pd.DataFrame(records)
    df_cmp["MAE_improve"] = (df_cmp["MAE_noSOC"] - df_cmp["MAE_SOC"]) / df_cmp["MAE_noSOC"]
    df_cmp["R2_improve"] = (df_cmp["R2_SOC"] - df_cmp["R2_noSOC"]) / df_cmp["R2_noSOC"]

    return df_cmp


def process_data(df_train_used, df_test, M):
    y_true, y_oracle = [], []
    for _, row in df_test.iterrows():
        try:
            y_true.append(row["SOH"])
            y_oracle.append(predict_soh_continuous(row, M, mode="oracle"))
        except Exception:
            continue

    soh_models_no_soc = train_soh_models_no_soc(df_train_used)

    def predict_soh_no_soc(row):
        U = row[FEATURES].values.reshape(1, -1)
        mat = row["Mat"]
        if mat not in soh_models_no_soc:
            return np.nan
        X_soh_scaled = soh_models_no_soc[mat]["scaler"].transform(U)
        return soh_models_no_soc[mat]["model"].predict(X_soh_scaled)[0]

    y_no_soc = []
    for _, row in df_test.iterrows():
        try:
            y_no_soc.append(predict_soh_no_soc(row))
        except Exception:
            y_no_soc.append(np.nan)
    y_no_soc = np.array(y_no_soc)

    y_t, y_n, y_s = np.array(y_true), y_no_soc, np.array(y_oracle)
    mask_no, mask_soc = ~np.isnan(y_n), ~np.isnan(y_s)
    print("=========== Overall SOH Performance ===========")
    print(f"SOC-unaware | MAE = {mean_absolute_error(y_t[mask_no], y_n[mask_no]):.4f} "
          f"| R2 = {r2_score(y_t[mask_no], y_n[mask_no]):.4f}")
    print(f"SOC-aware   | MAE = {mean_absolute_error(y_t[mask_soc], y_s[mask_soc]):.4f} "
          f"| R2 = {r2_score(y_t[mask_soc], y_s[mask_soc]):.4f}")
    df_cmp = compare_metrics_by_material(df_test, y_no_soc, y_oracle)
    print(df_cmp)

    soh = df_test[["Mat", "No.", "SOC", "SOH"]].reset_index(drop=True)
    soh["y_oracle"] = y_oracle
    soh["y_no_soc"] = y_no_soc
    return {"soh_predictions.csv": soh, "df_cmp.csv": df_cmp}
