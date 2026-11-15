"""Model generality of the RL-derived data requirement.

The cascaded pipeline (pulses -> material -> SOC -> SOH) is re-evaluated with
four model families (RF, ExtraTrees, SVM, MLP). For every number of
pulses (1-5) and training sample ratio (0.1-1.0), the pulse blocks are drawn with
ten seeds (10, 20, ..., 100); the same seed is used for training-sample
subsampling and for the models. Q-learning is applied to the mean performance
landscape and to the landscape of each seed.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR

from ..common import material_files

MATERIAL_ORDER = ["LFP", "LMO", "NMC"]

MODEL_NAMES = ["RF", "ExtraTrees", "SVM", "MLP"]

TEST_RATIO = 0.25
SPLIT_SEED = 42
FEATURE_SEEDS = list(range(10, 101, 10))

PULSE_FEATURES = {
    1: [f"U{i}" for i in range(1, 5)],
    2: [f"U{i}" for i in range(5, 9)],
    3: [f"U{i}" for i in range(9, 13)],
    4: [f"U{i}" for i in range(13, 17)],
    5: [f"U{i}" for i in range(17, 21)],
}

sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pulse_nums = [1, 2, 3, 4, 5]


def load_all_data():
    frames = []
    required_cols = ["No.", "SOC", "SOH"] + [f"U{i}" for i in range(1, 21)]

    for mat in MATERIAL_ORDER:
        for path in map(Path, material_files[mat]):
            df = pd.read_excel(path)
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"{path.name} is missing columns: {missing}")

            df = df.copy()
            df["Mat"] = mat
            df["BatteryKey"] = mat + "_" + df["No."].astype(str)
            frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    model_cols = ["Mat", "BatteryKey", "SOC", "SOH"] + [f"U{i}" for i in range(1, 21)]
    return df_all.dropna(subset=model_cols).reset_index(drop=True)


def battery_level_split(df, test_ratio=0.25, random_state=42):
    rng = np.random.default_rng(random_state)
    test_keys = []

    for mat in sorted(df["Mat"].unique()):
        keys = df.loc[df["Mat"] == mat, "BatteryKey"].drop_duplicates().to_numpy()
        if len(keys) < 2:
            raise ValueError(f"{mat} contains fewer than two unique batteries.")

        n_test = max(1, int(len(keys) * test_ratio))
        n_test = min(n_test, len(keys) - 1)
        selected = rng.choice(keys, size=n_test, replace=False)
        test_keys.extend(selected.tolist())

    mask = df["BatteryKey"].isin(test_keys)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def sample_pulse_features(pulse_num, seed=None):
    rng = np.random.default_rng(seed)
    pulses = sorted(
        rng.choice(
            list(PULSE_FEATURES.keys()),
            size=pulse_num,
            replace=False,
        ).tolist()
    )
    feats = [feature for pulse in pulses for feature in PULSE_FEATURES[pulse]]
    return feats, pulses


def build_models(model_name, random_state):
    """Classifier and SOC/SOH regressors of one model family."""
    if model_name == "RF":
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        soc_regressor = RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        soh_regressor = RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "ExtraTrees":
        classifier = ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        soc_regressor = ExtraTreesRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
        soh_regressor = ExtraTreesRegressor(
            n_estimators=300,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )

    elif model_name == "SVM":
        classifier = SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            random_state=random_state,
        )
        soc_regressor = SVR(
            C=10.0,
            epsilon=0.1,
            kernel="rbf",
            gamma="scale",
        )
        soh_regressor = SVR(
            C=10.0,
            epsilon=0.01,
            kernel="rbf",
            gamma="scale",
        )

    elif model_name == "MLP":
        classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        )
        soc_regressor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        )
        soh_regressor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"Unsupported model_name={model_name!r}. "
            "Choose from RF, ExtraTrees, SVM, or MLP."
        )

    return classifier, soc_regressor, soh_regressor


def evaluate_metrics(
    df_train,
    df_test,
    feature_cols,
    sample_ratio,
    model_name,
    random_state=0,
):
    """Cascaded pipeline: pulses -> material -> SOC -> SOH."""

    df_sub = df_train.sample(
        frac=sample_ratio,
        random_state=random_state,
    )

    X_mat = df_sub[feature_cols].values
    y_mat = df_sub["Mat"].values

    scaler_mat = StandardScaler()
    X_mat_s = scaler_mat.fit_transform(X_mat)

    le = LabelEncoder()
    y_mat_enc = le.fit_transform(y_mat)

    if len(le.classes_) < 3:
        raise ValueError(
            "The sampled training subset does not contain all three materials."
        )

    clf_mat, _, _ = build_models(model_name, random_state)
    clf_mat.fit(X_mat_s, y_mat_enc)

    soc_models = {}
    soh_models = {}

    for mat in df_sub["Mat"].unique():
        train_m = df_sub[df_sub["Mat"] == mat]
        if len(train_m) < 20:
            continue

        X = train_m[feature_cols].values
        y_soc = train_m["SOC"].values
        y_soh = train_m["SOH"].values

        scaler_soc = StandardScaler()
        X_s = scaler_soc.fit_transform(X)

        _, soc_model, soh_model = build_models(model_name, random_state)
        soc_model.fit(X_s, y_soc)

        soc_pred_train = soc_model.predict(X_s).reshape(-1, 1)
        X_soh = np.hstack([X, soc_pred_train])

        scaler_soh = StandardScaler()
        X_soh_s = scaler_soh.fit_transform(X_soh)
        soh_model.fit(X_soh_s, y_soh)

        soc_models[mat] = (soc_model, scaler_soc)
        soh_models[mat] = (soh_model, scaler_soh)

    U = df_test[feature_cols].values.astype(float)
    mat_pred = le.inverse_transform(clf_mat.predict(scaler_mat.transform(U)))

    soc_pred = np.full(len(U), np.nan)
    soh_pred = np.full(len(U), np.nan)
    for mat in soc_models:
        idx = mat_pred == mat
        if not idx.any():
            continue

        soc_model, scaler_soc = soc_models[mat]
        soc_pred[idx] = soc_model.predict(scaler_soc.transform(U[idx]))

        soh_model, scaler_soh = soh_models[mat]
        X_soh = np.hstack([U[idx], soc_pred[idx].reshape(-1, 1)])
        soh_pred[idx] = soh_model.predict(scaler_soh.transform(X_soh))

    valid = ~np.isnan(soh_pred)
    if not valid.any():
        raise RuntimeError("No valid test predictions were generated.")

    return {
        "soh_mae": mean_absolute_error(df_test["SOH"].values[valid], soh_pred[valid]),
        "soc_mae": (mean_absolute_error(df_test["SOC"].values[valid], soc_pred[valid])) / 100,
        "mat_acc": accuracy_score(df_test["Mat"].values[valid], mat_pred[valid]),
        "n_test": int(valid.sum()),
    }


def run_task_rl(
    H_perf,
    pulse_nums,
    sample_ratios,
    task_type="min",
    tol=0.01,
    cost_w_ratio=5.0,
    n_episodes=500,
    seed=0,
):
    np.random.seed(seed)
    random.seed(seed)

    if task_type == "min":
        best_val = np.nanmin(H_perf)
        mask_sufficient = H_perf <= best_val + tol
    elif task_type == "max":
        best_val = np.nanmax(H_perf)
        mask_sufficient = H_perf >= best_val - tol
    else:
        raise ValueError("task_type must be 'min' or 'max'.")

    cost = np.zeros_like(H_perf, dtype=float)
    for i, pulse_num in enumerate(pulse_nums):
        for j, ratio in enumerate(sample_ratios):
            cost[i, j] = pulse_num + cost_w_ratio * ratio

    def reward(i, j):
        if not mask_sufficient[i, j]:
            return -10.0
        return -cost[i, j] + 1.0

    n_p, n_r = H_perf.shape
    Q = np.zeros((n_p, n_r, 4), dtype=float)
    actions = {
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
                action = random.randint(0, 3)
            else:
                action = int(np.argmax(Q[i, j]))

            di, dj = actions[action]
            ni = int(np.clip(i + di, 0, n_p - 1))
            nj = int(np.clip(j + dj, 0, n_r - 1))

            current_reward = reward(ni, nj)
            Q[i, j, action] += alpha * (
                current_reward
                + gamma * np.max(Q[ni, nj])
                - Q[i, j, action]
            )
            i, j = ni, nj

    V = np.max(Q, axis=2)
    i_opt, j_opt = np.unravel_index(np.argmax(V), V.shape)

    return {
        "pulse_num": pulse_nums[i_opt],
        "train_ratio": sample_ratios[j_opt],
        "selected_performance": float(H_perf[i_opt, j_opt]),
        "best_performance": float(best_val),
        "mask_sufficient": mask_sufficient,
        "cost": cost,
        "Q": Q,
        "V": V,
    }


def dataset_summary(df):
    records = []
    for mat in sorted(df["Mat"].unique()):
        group = df[df["Mat"] == mat]
        records.append({
            "material": mat,
            "samples": len(group),
            "batteries": group["BatteryKey"].nunique(),
            "soc_min": group["SOC"].min(),
            "soc_max": group["SOC"].max(),
            "soh_min": group["SOH"].min(),
            "soh_max": group["SOH"].max(),
        })
    return pd.DataFrame(records)


def evaluate_model(model_name, df_train_used, df_test):
    """Grid of one model family over all feature seeds."""
    seed_matrices = {
        seed: {
            "soh": np.zeros((len(pulse_nums), len(sample_ratios))),
            "soc": np.zeros((len(pulse_nums), len(sample_ratios))),
            "mat": np.zeros((len(pulse_nums), len(sample_ratios))),
        }
        for seed in FEATURE_SEEDS
    }
    raw_records = []

    for seed in FEATURE_SEEDS:
        for i, pulse_num in enumerate(pulse_nums):
            feature_cols, selected_pulses = sample_pulse_features(pulse_num, seed=seed)

            for j, ratio in enumerate(sample_ratios):
                result = evaluate_metrics(
                    df_train_used,
                    df_test,
                    feature_cols,
                    ratio,
                    model_name=model_name,
                    random_state=seed,
                )

                seed_matrices[seed]["soh"][i, j] = result["soh_mae"]
                seed_matrices[seed]["soc"][i, j] = result["soc_mae"]
                seed_matrices[seed]["mat"][i, j] = result["mat_acc"]

                raw_records.append({
                    "model": model_name,
                    "feature_seed": seed,
                    "pulse_num": pulse_num,
                    "selected_pulses": ",".join(map(str, selected_pulses)),
                    "selected_features": ",".join(feature_cols),
                    "train_ratio": ratio,
                    **result,
                })

                print(
                    f"{model_name:>10} | seed={seed:>3} | "
                    f"pulses={pulse_num} | ratio={ratio:.1f} | "
                    f"selected={selected_pulses} | "
                    f"SOH={result['soh_mae']:.4f} | "
                    f"SOC={result['soc_mae']:.4f} | "
                    f"MAT={result['mat_acc']:.4f}",
                    flush=True,
                )

    return seed_matrices, raw_records


def process_data(model_names=MODEL_NAMES):
    df = load_all_data()
    df_train_used, df_test = battery_level_split(
        df,
        test_ratio=TEST_RATIO,
        random_state=SPLIT_SEED,
    )
    print(f"train samples = {len(df_train_used)}, test samples = {len(df_test)}")

    out = {"dataset_summary.csv": dataset_summary(df)}
    all_raw, all_rl_mean, all_rl_seed = [], [], []

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        seed_matrices, raw_records = evaluate_model(model_name, df_train_used, df_test)
        all_raw.extend(raw_records)

        stacks = {
            key: np.stack([seed_matrices[s][key] for s in FEATURE_SEEDS])
            for key in ["soh", "soc", "mat"]
        }
        H_mean = {key: np.mean(v, axis=0) for key, v in stacks.items()}
        H_std = {key: np.std(v, axis=0, ddof=1) for key, v in stacks.items()}

        for name, matrix in [
            ("SOH_MAE_mean", H_mean["soh"]),
            ("SOC_MAE_mean", H_mean["soc"]),
            ("Material_Accuracy_mean", H_mean["mat"]),
            ("SOH_MAE_std", H_std["soh"]),
            ("SOC_MAE_std", H_std["soc"]),
            ("Material_Accuracy_std", H_std["mat"]),
        ]:
            out[f"{model_name}/{name}.csv"] = pd.DataFrame(
                matrix, index=pulse_nums, columns=sample_ratios
            )

        res_soh = run_task_rl(H_mean["soh"], pulse_nums, sample_ratios,
                              task_type="min", tol=0.01, seed=0)
        res_soc = run_task_rl(H_mean["soc"], pulse_nums, sample_ratios,
                              task_type="min", tol=0.01, seed=0)
        res_mat = run_task_rl(H_mean["mat"], pulse_nums, sample_ratios,
                              task_type="max", tol=0.02, seed=0)

        for task, result in [("SOH", res_soh), ("SOC", res_soc), ("Material", res_mat)]:
            all_rl_mean.append({
                "Model": model_name,
                "Task": task,
                "Optimal Number of Pulses": result["pulse_num"],
                "Optimal Training Ratio": result["train_ratio"],
            })
            print(f"{model_name} | {task}: pulse={result['pulse_num']}, "
                  f"ratio={result['train_ratio']}, "
                  f"performance={result['selected_performance']:.4f}")

        for seed in FEATURE_SEEDS:
            matrices = seed_matrices[seed]
            seed_results = {
                "SOH": run_task_rl(matrices["soh"], pulse_nums, sample_ratios,
                                   "min", 0.01, seed=seed),
                "SOC": run_task_rl(matrices["soc"], pulse_nums, sample_ratios,
                                   "min", 0.01, seed=seed),
                "Material": run_task_rl(matrices["mat"], pulse_nums, sample_ratios,
                                        "max", 0.02, seed=seed),
            }
            for task, result in seed_results.items():
                all_rl_seed.append({
                    "model": model_name,
                    "feature_seed": seed,
                    "task": task,
                    "optimal_pulse_num": result["pulse_num"],
                    "optimal_train_ratio": result["train_ratio"],
                    "selected_performance": result["selected_performance"],
                    "best_performance": result["best_performance"],
                    "selected_cost": result["pulse_num"] + 5.0 * result["train_ratio"],
                })

    df_rl_seed = pd.DataFrame(all_rl_seed)
    frequency = (
        df_rl_seed
        .groupby(["model", "task", "optimal_pulse_num", "optimal_train_ratio"])
        .size()
        .reset_index(name="selection_count")
    )
    frequency["selection_frequency"] = frequency["selection_count"] / len(FEATURE_SEEDS)

    out.update({
        "raw_results.csv": pd.DataFrame(all_raw),
        "RL_optimal_data_requirement.csv": pd.DataFrame(all_rl_mean),
        "RL_optimal_for_each_seed.csv": df_rl_seed,
        "RL_configuration_frequency.csv": frequency,
    })
    return out
