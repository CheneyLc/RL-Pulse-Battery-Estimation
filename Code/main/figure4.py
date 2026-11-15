"""Figure 4: SOC estimation with the per-material SOC models (true material)."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from ..common import FEATURES


def process_data(df_test, M):
    soc_models = M["soc_models"]

    soc_true = []
    soc_pred_oracle = []

    for _, row in df_test.iterrows():

        U = row[FEATURES].values.reshape(1, -1)
        true_soc = row["SOC"]
        soc_true.append(true_soc)

        mat_true = row["Mat"]

        if mat_true in soc_models:
            soc_scaler = soc_models[mat_true]["scaler"]
            soc_model = soc_models[mat_true]["model"]

            U_soc_scaled = soc_scaler.transform(U)
            soc_o = soc_model.predict(U_soc_scaled)[0]
        else:
            soc_o = np.nan

        soc_pred_oracle.append(soc_o)

    soc_true = np.array(soc_true)
    soc_pred_oracle = np.array(soc_pred_oracle)

    mask = ~np.isnan(soc_pred_oracle)
    print(f"MAE = {mean_absolute_error(soc_true[mask], soc_pred_oracle[mask]):.2f}%")
    print(f"R² = {r2_score(soc_true[mask], soc_pred_oracle[mask]):.4f}")

    df_soc = pd.DataFrame({
        "SOC_true": soc_true,
        "SOC_pred": soc_pred_oracle,
        "Mat": df_test["Mat"].values
    })
    return {"soc_predictions.csv": df_soc}
