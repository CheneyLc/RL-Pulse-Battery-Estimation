"""Figure 3: material classification, data-sufficiency grids (number of pulses and
positive/negative pulse combinations x training sample ratio) and the Q-learning
search for the minimal data requirement."""
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from ..common import FEATURES

PULSE_FEATURES = {
    1: [f"U{i}" for i in range(1, 5)],
    2: [f"U{i}" for i in range(5, 9)],
    3: [f"U{i}" for i in range(9, 13)],
    4: [f"U{i}" for i in range(13, 17)],
    5: [f"U{i}" for i in range(17, 21)],
}
sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pulse_nums = [1, 2, 3, 4, 5]

POS_PULSES = {
    1: [f"U{i}" for i in range(1, 5)],
    2: [f"U{i}" for i in range(9, 13)],
    3: [f"U{i}" for i in range(17, 21)],
}

NEG_PULSES = {
    1: [f"U{i}" for i in range(5, 9)],
    2: [f"U{i}" for i in range(13, 17)],
}
POS_COUNTS = [0, 1, 2, 3]
NEG_COUNTS = [0, 1, 2]

COMBOS = []
for np_ in POS_COUNTS:
    for nn_ in NEG_COUNTS:
        if np_ == 0 and nn_ == 0:
            continue
        COMBOS.append((np_, nn_))


def material_predictions(df_test, M):
    scaler, clf_mat, le_mat = M["scaler"], M["clf_mat"], M["le_mat"]
    X_test = df_test[FEATURES].values
    X_test_scaled = scaler.transform(X_test)

    y_mat_true = df_test["Mat"].values
    y_mat_pred_enc = clf_mat.predict(X_test_scaled)
    y_mat_pred = le_mat.inverse_transform(y_mat_pred_enc)

    print("Material classification accuracy:",
          accuracy_score(y_mat_true, y_mat_pred))
    return pd.DataFrame({"y_mat_true": y_mat_true, "y_mat_pred": y_mat_pred})


def sample_pulse_features(pulse_num, seed=None):
    rng = np.random.default_rng(seed)
    pulses = rng.choice(list(PULSE_FEATURES.keys()),
                        size=pulse_num,
                        replace=False)
    feats = []
    for p in pulses:
        feats.extend(PULSE_FEATURES[p])
    return feats


def evaluate_metrics(
    df_train,
    df_test,
    feature_cols,
    sample_ratio,
    random_state=0
):

    df_sub = df_train.sample(frac=sample_ratio, random_state=random_state)

    X_mat = df_sub[feature_cols].values
    y_mat = df_sub["Mat"].values

    scaler_mat = StandardScaler()
    X_mat_s = scaler_mat.fit_transform(X_mat)

    le = LabelEncoder()
    y_mat_enc = le.fit_transform(y_mat)

    clf_mat = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=0
    )
    clf_mat.fit(X_mat_s, y_mat_enc)

    soc_models, soh_models = {}, {}

    for mat in df_sub["Mat"].unique():

        train_m = df_sub[df_sub["Mat"] == mat]
        if len(train_m) < 20:
            continue

        X = train_m[feature_cols].values
        y_soc = train_m["SOC"].values
        y_soh = train_m["SOH"].values

        scaler_soc = StandardScaler()
        X_s = scaler_soc.fit_transform(X)

        soc_model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=0
        )
        soc_model.fit(X_s, y_soc)

        soc_pred = soc_model.predict(X_s).reshape(-1, 1)

        X_soh = np.hstack([X, soc_pred])
        scaler_soh = StandardScaler()
        X_soh_s = scaler_soh.fit_transform(X_soh)

        soh_model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            random_state=0
        )
        soh_model.fit(X_soh_s, y_soh)

        soc_models[mat] = (soc_model, scaler_soc)
        soh_models[mat] = (soh_model, scaler_soh)

    # Test samples are predicted in one batch per predicted material; each sample
    # goes through the same cascade (material -> SOC -> SOH) as a row-by-row loop.
    U = df_test[feature_cols].values.astype(float)
    mat_pred_all = le.inverse_transform(clf_mat.predict(scaler_mat.transform(U)))
    soc_pred_all = np.full(len(U), np.nan)
    soh_pred_all = np.full(len(U), np.nan)
    for mat, (soc_model, scaler_soc) in soc_models.items():
        idx = mat_pred_all == mat
        if not idx.any():
            continue
        soc_pred = soc_model.predict(scaler_soc.transform(U[idx]))
        soh_model, scaler_soh = soh_models[mat]
        X_soh = np.hstack([U[idx], soc_pred.reshape(-1, 1)])
        soc_pred_all[idx] = soc_pred
        soh_pred_all[idx] = soh_model.predict(scaler_soh.transform(X_soh))
    keep = np.isin(mat_pred_all, list(soc_models))

    y_mat_true = df_test["Mat"].values[keep]
    y_mat_pred = mat_pred_all[keep]
    y_soc_true = df_test["SOC"].values[keep]
    y_soc_pred = soc_pred_all[keep]
    y_soh_true = df_test["SOH"].values[keep]
    y_soh_pred = soh_pred_all[keep]

    return {
        "soh_mae": mean_absolute_error(y_soh_true, y_soh_pred),
        "soc_mae": mean_absolute_error(y_soc_true, y_soc_pred),
        "mat_acc": accuracy_score(y_mat_true, y_mat_pred),
    }


def pulse_count_grid(df_train_used, df_test):
    H_soh = np.zeros((len(pulse_nums), len(sample_ratios)))
    H_soc = np.zeros_like(H_soh)
    H_mat = np.zeros_like(H_soh)

    for i, pnum in enumerate(pulse_nums):
        for j, ratio in enumerate(sample_ratios):

            res = []
            for r in range(3):
                feats = sample_pulse_features(pnum, seed=r)
                res.append(
                    evaluate_metrics(
                        df_train_used,
                        df_test,
                        feats,
                        ratio,
                        random_state=r
                    )
                )

            H_soh[i, j] = np.mean([x["soh_mae"] for x in res])
            H_soc[i, j] = np.mean([x["soc_mae"] for x in res])
            H_mat[i, j] = np.mean([x["mat_acc"] for x in res])
            print(f"pulses={pnum} | ratio={ratio:.2f} | SOH={H_soh[i, j]:.4f}, "
                  f"SOC={H_soc[i, j]:.4f}, MAT={H_mat[i, j]:.3f}", flush=True)
    return H_soh, H_soc, H_mat


def summary_across_rows(H_mat, H_soc, H_soh, label):
    mat_mean = np.mean(H_mat, axis=0)
    mat_std = np.std(H_mat, axis=0, ddof=1)
    soc_mean = np.mean(H_soc, axis=0)
    soc_std = np.std(H_soc, axis=0, ddof=1)
    soh_mean = np.mean(H_soh, axis=0)
    soh_std = np.std(H_soh, axis=0, ddof=1)

    df_summary = pd.DataFrame({
        "Training sample ratio": sample_ratios,

        "Material Accuracy Mean": mat_mean,
        "Material Accuracy Std": mat_std,

        "SOC MAE Mean": soc_mean,
        "SOC MAE Std": soc_std,

        "SOH MAE Mean": soh_mean,
        "SOH MAE Std": soh_std,
    })
    print(f"\nMean and standard deviation across {label}:")
    print(df_summary.to_string(index=False))

    df_summary_display = pd.DataFrame({
        "Training sample ratio": sample_ratios,
        "Material Accuracy": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mat_mean, mat_std)],
        "SOC MAE": [f"{m:.4f} ± {s:.4f}" for m, s in zip(soc_mean, soc_std)],
        "SOH MAE": [f"{m:.4f} ± {s:.4f}" for m, s in zip(soh_mean, soh_std)],
    })
    print("\nPublication-style table:")
    print(df_summary_display.to_string(index=False))
    return df_summary, df_summary_display


def run_task_rl(
    H_perf,
    pulse_nums,
    sample_ratios,
    task_type="min",
    tol=0.01,
    cost_w_ratio=5.0,
    n_episodes=500,
    seed=0
):

    np.random.seed(seed)
    random.seed(seed)

    if task_type == "min":
        best_val = np.nanmin(H_perf)
        mask_sufficient = H_perf <= best_val + tol
    else:
        best_val = np.nanmax(H_perf)
        mask_sufficient = H_perf >= best_val - tol

    cost = np.zeros_like(H_perf)
    for i, p in enumerate(pulse_nums):
        for j, r in enumerate(sample_ratios):
            cost[i, j] = p + cost_w_ratio * r

    def reward(i, j):
        if not mask_sufficient[i, j]:
            return -10.0
        return -cost[i, j] + 1.0

    n_p, n_r = H_perf.shape
    Q = np.zeros((n_p, n_r, 4))

    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    alpha, gamma, eps = 0.2, 0.9, 0.2

    for _ in range(n_episodes):
        i = random.randint(0, n_p - 1)
        j = random.randint(0, n_r - 1)

        for _ in range(20):
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                a = np.argmax(Q[i, j])

            di, dj = ACTIONS[a]
            ni = np.clip(i + di, 0, n_p - 1)
            nj = np.clip(j + dj, 0, n_r - 1)

            r = reward(ni, nj)
            Q[i, j, a] += alpha * (
                r + gamma * np.max(Q[ni, nj]) - Q[i, j, a]
            )

            i, j = ni, nj

    V = np.max(Q, axis=2)
    i_opt, j_opt = np.unravel_index(np.argmax(V), V.shape)

    return {
        "pulse_num": pulse_nums[i_opt],
        "train_ratio": sample_ratios[j_opt],
        "mask_sufficient": mask_sufficient,
        "cost": cost,
        "Q": Q,
        "V": V,
    }


def rl_optimum(H_soh, H_soc, H_mat):
    res_soh = run_task_rl(H_perf=H_soh, pulse_nums=pulse_nums, sample_ratios=sample_ratios,
                          task_type="min", tol=0.01)
    res_soc = run_task_rl(H_perf=H_soc, pulse_nums=pulse_nums, sample_ratios=sample_ratios,
                          task_type="min", tol=0.01)
    res_mat = run_task_rl(H_perf=H_mat, pulse_nums=pulse_nums, sample_ratios=sample_ratios,
                          task_type="max", tol=0.02)
    print("Optimal data sufficiency per task")
    print("----------------------------------")
    print(f"SOH  : pulse={res_soh['pulse_num']}, ratio={res_soh['train_ratio']}")
    print(f"SOC  : pulse={res_soc['pulse_num']}, ratio={res_soc['train_ratio']}")
    print(f"MAT  : pulse={res_mat['pulse_num']}, ratio={res_mat['train_ratio']}")
    table = pd.DataFrame([
        {"Task": t, "task_type": tt, "tol": tol, "pulse_num": r["pulse_num"],
         "train_ratio": r["train_ratio"]}
        for t, tt, tol, r in [("SOH", "min", 0.01, res_soh), ("SOC", "min", 0.01, res_soc),
                              ("MAT", "max", 0.02, res_mat)]])
    V = {t: pd.DataFrame(r["V"], index=pulse_nums, columns=sample_ratios)
         for t, r in [("SOH", res_soh), ("SOC", res_soc), ("MAT", res_mat)]}
    return table, V


def sample_pos_neg_features(n_pos, n_neg, seed=None):
    rng = np.random.default_rng(seed)

    pos_keys = list(POS_PULSES.keys())
    neg_keys = list(NEG_PULSES.keys())

    if n_pos > len(pos_keys) or n_neg > len(neg_keys):
        return None
    if n_pos == 0 and n_neg == 0:
        return None

    feats = []

    if n_pos > 0:
        pos_sel = rng.choice(pos_keys, size=n_pos, replace=False)
        for k in pos_sel:
            feats.extend(POS_PULSES[k])

    if n_neg > 0:
        neg_sel = rng.choice(neg_keys, size=n_neg, replace=False)
        for k in neg_sel:
            feats.extend(NEG_PULSES[k])

    return feats


def pos_neg_grid(df_train_used, df_test):
    H_soh = np.zeros((len(COMBOS), len(sample_ratios)))
    H_soc = np.zeros_like(H_soh)
    H_mat = np.zeros_like(H_soh)

    for i, (n_pos, n_neg) in enumerate(COMBOS):
        for j, ratio in enumerate(sample_ratios):

            res = []
            for r in range(3):
                feats = sample_pos_neg_features(n_pos, n_neg, seed=r)
                if feats is None:
                    continue

                res.append(
                    evaluate_metrics(
                        df_train_used,
                        df_test,
                        feats,
                        ratio,
                        random_state=r
                    )
                )

            H_soh[i, j] = np.mean([x["soh_mae"] for x in res])
            H_soc[i, j] = np.mean([x["soc_mae"] for x in res])
            H_mat[i, j] = np.mean([x["mat_acc"] for x in res])

            print(f"+{n_pos}/-{n_neg} | ratio={ratio:.2f} | "
                  f"SOH={H_soh[i, j]:.4f}, "
                  f"SOC={H_soc[i, j]:.4f}, "
                  f"MAT={H_mat[i, j]:.3f}", flush=True)
    return H_soh, H_soc, H_mat


def _matrix(H, index):
    return pd.DataFrame(H, index=index, columns=sample_ratios)


def process_data(df_train_used, df_test, M):
    out = {"material_predictions.csv": material_predictions(df_test, M),
           "classes.csv": pd.DataFrame({"le_mat_classes_": M["le_mat"].classes_})}

    H_soh, H_soc, H_mat = pulse_count_grid(df_train_used, df_test)
    s, sd = summary_across_rows(H_mat, H_soc, H_soh, "pulse numbers")
    rl, V = rl_optimum(H_soh, H_soc, H_mat)
    out.update({"pulse_count_H_soh.csv": _matrix(H_soh, pulse_nums),
                "pulse_count_H_soc.csv": _matrix(H_soc, pulse_nums),
                "pulse_count_H_mat.csv": _matrix(H_mat, pulse_nums),
                "pulse_count_summary.csv": s, "pulse_count_summary_display.csv": sd,
                "rl_optimum.csv": rl,
                **{f"rl_V_{t}.csv": v for t, v in V.items()}})

    labels = [f"+{p} / -{n}" for p, n in COMBOS]
    H_soh, H_soc, H_mat = pos_neg_grid(df_train_used, df_test)
    s, sd = summary_across_rows(H_mat, H_soc, H_soh, "pulse combinations")
    out.update({"pos_neg_H_soh.csv": _matrix(H_soh, labels),
                "pos_neg_H_soc.csv": _matrix(H_soc, labels),
                "pos_neg_H_mat.csv": _matrix(H_mat, labels),
                "pos_neg_summary.csv": s, "pos_neg_summary_display.csv": sd})
    return out
