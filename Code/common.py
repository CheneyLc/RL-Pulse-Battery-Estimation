"""Data loading and the full-feature models
(material classifier, per-material SOC and SOH estimators)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Pulse-test workbooks (from Zenodo): GitHub/data/, or a sibling Zenodo/data/ folder.
_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = next((p for p in [_ROOT / "data", _ROOT.parent / "Zenodo" / "data"] if p.is_dir()),
                _ROOT / "data")

DATASETS = {
    1: {"cells": {"NMC": "NMC_2.1Ah", "LMO": "LMO_10Ah", "LFP": "LFP_35Ah"},
        "widths": [5000]},
    2: {"cells": {"NMC": "NMC_21Ah", "LMO": "LMO_24Ah", "LFP": "LFP_20Ah"},
        "widths": [30, 50, 70, 100, 300, 500, 700, 1000, 3000]},
}

material_files = {}


def use_dataset(dataset=1):
    """Select the data set (1 or 2) used by all computations.

    Dataset 1: one file per material (pulse width 5,000 ms).
    Dataset 2: nine files per material (pulse widths 30-3,000 ms), pooled.
    Returns the name of the result folder ("dataset1" or "dataset2").
    """
    spec = DATASETS[dataset]
    material_files.clear()
    for mat, cell in spec["cells"].items():
        material_files[mat] = [str(DATA_DIR / f"dataset{dataset}" / f"{cell}_W_{w}.xlsx")
                               for w in spec["widths"]]
    return f"dataset{dataset}"


use_dataset(1)

FEATURES = [f"U{i}" for i in range(1, 21)]


def load_all_data(material_files):
    dfs = []
    for mat, files in material_files.items():
        for f in files:
            df = pd.read_excel(f)
            df["Mat"] = mat
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def split_train_test(df):
    TEST_RATIO = 0.25
    np.random.seed(42)

    test_mask = np.zeros(len(df), dtype=bool)

    for mat in df["Mat"].unique():
        mat_mask = df["Mat"] == mat
        battery_ids = df.loc[mat_mask, "No."].unique()

        test_ids = np.random.choice(
            battery_ids,
            size=max(1, int(len(battery_ids) * TEST_RATIO)),
            replace=False
        )

        test_mask |= df["No."].isin(test_ids).values

    df_test = df[test_mask].copy()
    df_train_used = df[~test_mask].copy()
    return df_train_used, df_test


def train_material_classifier(df_train_used):
    X_train = df_train_used[FEATURES].values
    y_mat = df_train_used["Mat"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    le_mat = LabelEncoder()
    y_mat_enc = le_mat.fit_transform(y_mat)

    clf_mat = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
    clf_mat.fit(X_train_scaled, y_mat_enc)
    return scaler, le_mat, clf_mat


def train_soh_models(df_train_used):
    """Per-material SOH estimators with SOC as an additional input."""
    soh_models = {}

    for mat in df_train_used["Mat"].unique():

        idx = df_train_used["Mat"] == mat

        X_m = df_train_used.loc[idx, FEATURES].values
        SOC_m = df_train_used.loc[idx, "SOC"].values.reshape(-1, 1)
        X_m = np.hstack([X_m, SOC_m])

        y_m = df_train_used.loc[idx, "SOH"].values

        scaler_soh = StandardScaler()
        X_m_scaled = scaler_soh.fit_transform(X_m)

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

        soh_models[mat] = {
            "model": model,
            "scaler": scaler_soh
        }

        print(f"SOH model trained for {mat}, samples={len(y_m)}")
    return soh_models


def train_soc_models(df_train_used):
    """Per-material SOC estimators."""
    soc_models = {}

    for mat in df_train_used["Mat"].unique():

        idx = df_train_used["Mat"] == mat
        X_m = df_train_used.loc[idx, FEATURES].values
        y_soc = df_train_used.loc[idx, "SOC"].values

        scaler_soc = StandardScaler()
        X_m_scaled = scaler_soc.fit_transform(X_m)

        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
        model.fit(X_m_scaled, y_soc)

        soc_models[mat] = {
            "model": model,
            "scaler": scaler_soc
        }

        print(f"SOC model trained for {mat}, samples={len(y_soc)}")
    return soc_models


def build_full_models(df_train_used):
    scaler, le_mat, clf_mat = train_material_classifier(df_train_used)
    soh_models = train_soh_models(df_train_used)
    soc_models = train_soc_models(df_train_used)
    return {"scaler": scaler, "le_mat": le_mat, "clf_mat": clf_mat,
            "soh_models": soh_models, "soc_models": soc_models}


def predict_soh_continuous(row, M, mode="soft"):
    """Cascaded SOH estimation for one sample.

    mode="oracle": true material and true SOC;
    otherwise:     predicted material and predicted SOC.
    """
    scaler, clf_mat, le_mat = M["scaler"], M["clf_mat"], M["le_mat"]
    soc_models, soh_models = M["soc_models"], M["soh_models"]

    U = row[FEATURES].values.reshape(1, -1)
    U_scaled_for_clf = scaler.transform(U)

    if mode == "oracle":
        mat = row["Mat"]
    else:
        mat_id = clf_mat.predict(U_scaled_for_clf)[0]
        mat = le_mat.inverse_transform([mat_id])[0]

    if mat not in soc_models or mat not in soh_models:
        return np.nan

    curr_soc_model = soc_models[mat]["model"]
    curr_soc_scaler = soc_models[mat]["scaler"]

    if mode == "oracle":
        soc = row["SOC"]
    else:
        U_scaled_for_soc = curr_soc_scaler.transform(U)
        soc = curr_soc_model.predict(U_scaled_for_soc)[0]

    curr_soh_model = soh_models[mat]["model"]
    curr_soh_scaler = soh_models[mat]["scaler"]

    X_soh = np.hstack([U, [[soc]]])
    X_soh_scaled = curr_soh_scaler.transform(X_soh)

    return curr_soh_model.predict(X_soh_scaled)[0]
