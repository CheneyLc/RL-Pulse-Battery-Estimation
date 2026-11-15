"""Figures 3d, 5j and 5k: mean kNN density of the test samples with respect to the
training samples and the SOH error, for k pulse blocks x battery-level training ratio."""
import random

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def compute_knn_density(X_train, X_query, k=10):
    k = min(k, len(X_train) - 1)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train)
    dists, _ = nbrs.kneighbors(X_query)
    return 1.0 / (np.mean(dists, axis=1) + 1e-8)


def train_soc_continuous_pipeline(df_train, U_cols):

    X_tr = df_train[U_cols].values
    scaler_X = StandardScaler()
    X_tr_scaled = scaler_X.fit_transform(X_tr)

    le_mat = LabelEncoder()
    y_mat = le_mat.fit_transform(df_train["Mat"])

    clf_mat = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42
    )
    clf_mat.fit(X_tr_scaled, y_mat)

    soc_models = {}

    for mat in df_train["Mat"].unique():
        idx = df_train["Mat"] == mat

        X_m = X_tr_scaled[idx]
        y_soc = df_train.loc[idx, "SOC"].values

        reg = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=42
        )
        reg.fit(X_m, y_soc)

        soc_models[mat] = reg

    soh_models = {}

    for mat in df_train["Mat"].unique():
        idx = df_train["Mat"] == mat

        X_u = df_train.loc[idx, U_cols].values
        soc = df_train.loc[idx, "SOC"].values.reshape(-1, 1)
        X_soh = np.hstack([X_u, soc])
        y_soh = df_train.loc[idx, "SOH"].values

        scaler_soh = StandardScaler()
        X_soh_scaled = scaler_soh.fit_transform(X_soh)

        reg = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )
        reg.fit(X_soh_scaled, y_soh)

        soh_models[mat] = {
            "model": reg,
            "scaler": scaler_soh
        }

    return scaler_X, clf_mat, le_mat, soc_models, soh_models


def predict_soh_continuous(
    row, U_cols,
    scaler_X, clf_mat, le_mat,
    soc_models, soh_models,
    mode="oracle"
):
    U = row[U_cols].values.reshape(1, -1)
    U_scaled = scaler_X.transform(U)

    if mode == "oracle":
        mat = row["Mat"]
    else:
        mat_id = clf_mat.predict(U_scaled)[0]
        mat = le_mat.inverse_transform([mat_id])[0]

    if mat not in soc_models or mat not in soh_models:
        return np.nan

    if mode == "oracle":
        soc = row["SOC"]
    else:
        soc = soc_models[mat].predict(U_scaled)[0]

    X_soh = np.hstack([U, [[soc]]])
    X_soh_scaled = soh_models[mat]["scaler"].transform(X_soh)

    return soh_models[mat]["model"].predict(X_soh_scaled)[0]


def sample_training_batteries(df_train, ratio=1.0, seed=42):
    rng = np.random.RandomState(seed)
    dfs = []

    for mat in df_train["Mat"].unique():
        df_m = df_train[df_train["Mat"] == mat]
        battery_ids = df_m["No."].unique()

        n_sel = max(1, int(len(battery_ids) * ratio))
        sel_ids = rng.choice(battery_ids, size=n_sel, replace=False)

        dfs.append(df_m[df_m["No."].isin(sel_ids)])

    return pd.concat(dfs, ignore_index=True)


def sample_feature_blocks(blocks, k, seed=42):
    rng = random.Random(seed)
    keys = list(blocks.keys())
    sel_keys = rng.sample(keys, k)
    U_cols = []
    for kk in sel_keys:
        U_cols.extend(blocks[kk])
    return U_cols, sel_keys


FEATURE_BLOCKS = {
    "B1": [f"U{i}" for i in range(1, 5)],
    "B2": [f"U{i}" for i in range(5, 9)],
    "B3": [f"U{i}" for i in range(9, 13)],
    "B4": [f"U{i}" for i in range(13, 17)],
    "B5": [f"U{i}" for i in range(17, 21)],
}


def process_data(df_train_used, df_test):
    results_soh = []
    TRAIN_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for k in range(1, 6):
        for ratio in TRAIN_RATIOS:

            print(f"blocks={k}, train_ratio={ratio}", flush=True)

            df_sub = sample_training_batteries(
                df_train_used, ratio=ratio, seed=42
            )
            if len(df_sub) < 80:
                continue

            U_cols, sel_blocks = sample_feature_blocks(
                FEATURE_BLOCKS, k, seed=42
            )

            scaler_X, clf_mat, le_mat, soc_models, soh_models = \
                train_soc_continuous_pipeline(df_sub, U_cols)

            X_tr_scaled = scaler_X.transform(df_sub[U_cols].values)
            X_te_scaled = scaler_X.transform(df_test[U_cols].values)

            densities = compute_knn_density(X_tr_scaled, X_te_scaled)
            mean_density = np.nanmean(densities)

            # Oracle mode (true material and true SOC), predicted in one batch per
            # material; identical to calling predict_soh_continuous row by row.
            U = df_test[U_cols].values.astype(float)
            pred = np.full(len(U), np.nan)
            for mat in df_test["Mat"].unique():
                if mat not in soc_models or mat not in soh_models:
                    continue
                idx = (df_test["Mat"] == mat).values
                X_soh = np.hstack([U[idx], df_test["SOC"].values[idx].reshape(-1, 1)])
                X_soh_scaled = soh_models[mat]["scaler"].transform(X_soh)
                pred[idx] = soh_models[mat]["model"].predict(X_soh_scaled)
            ok = ~np.isnan(pred)
            y_true = df_test["SOH"].values[ok]
            y_pred = pred[ok]

            if len(y_pred) < 30:
                mae, r2 = np.nan, np.nan
            else:
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

            results_soh.append({
                "n_blocks": k,
                "train_ratio": ratio,
                "mean_density": mean_density,
                "soh_mae": mae,
                "soh_r2": r2,
            })

    return {"df_res.csv": pd.DataFrame(results_soh)}
