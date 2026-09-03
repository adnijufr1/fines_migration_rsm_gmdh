import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TRAIN_PATH = "MASTER_Train_Data_Hybrid.csv"
TEST_PATH = "MASTER_Test_Data_Hybrid.csv"

FEATURE_COLS = [
    "Interstitial Velocity, m/s",
    "Porosity",
    "Salt Conc. (PPM)",
    "PV",
    "Clay Percentage",
    "Zeta Clay (mV)",
    "Zeta Sand (mV)",
]
TARGET_COL = "Norm K"

POLY_DEGREE = 3
RIDGE_ALPHA = 0.001
N_CV_FOLDS = 5
RANDOM_STATE_CV = 48

RATIOS = [0.60, 0.70, 0.80]
RATIO_SEEDS = [35, 36, 37, 38, 39]


def aape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "AAPE": aape(y_true, y_pred),
    }


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=POLY_DEGREE, include_bias=True)),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[TARGET_COL].values
    return X_train, y_train, X_test, y_test, train_df, test_df


def run_holdout(X_train, y_train, X_test, y_test):
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    train_metrics = compute_metrics(y_train, pipe.predict(X_train))
    test_metrics = compute_metrics(y_test, pipe.predict(X_test))

    print("\nHoldout results")
    print("Train:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test: ", {k: round(v, 4) for k, v in test_metrics.items()})
    return pipe, train_metrics, test_metrics


def run_cv(X_train, y_train):
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE_CV)
    rows = []
    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        pipe = build_pipeline()
        pipe.fit(X_train[tr_idx], y_train[tr_idx])
        preds = pipe.predict(X_train[val_idx])
        m = compute_metrics(y_train[val_idx], preds)
        rows.append({"Fold": fold_i, "N": len(val_idx), **m})

    table = pd.DataFrame(rows)
    summary = table[["R2", "RMSE", "MAE", "AAPE"]].agg(["mean", "std"])

    print("\n5-fold CV results")
    print(table.round(4).to_string(index=False))
    print("Mean:", summary.loc["mean"].round(4).to_dict())
    print("SD:  ", summary.loc["std"].round(4).to_dict())
    return table, summary


def run_ratio_sensitivity(X, y):
    rows = []
    for ratio in RATIOS:
        metrics = {"R2": [], "RMSE": [], "MAE": [], "AAPE": []}
        for seed in RATIO_SEEDS:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=ratio, random_state=seed
            )
            pipe = build_pipeline()
            pipe.fit(X_train, y_train)
            m = compute_metrics(y_test, pipe.predict(X_test))
            for k in metrics:
                metrics[k].append(m[k])

        row = {"Ratio": f"{round(ratio*100)}:{round((1-ratio)*100)}"}
        for k, vals in metrics.items():
            row[f"{k}_mean"] = np.mean(vals)
            row[f"{k}_sd"] = np.std(vals, ddof=1)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\nRatio sensitivity")
    print(table.round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, train_df, test_df = load_data()

    run_holdout(X_train, y_train, X_test, y_test)
    run_cv(X_train, y_train)

    pooled_df = pd.concat([train_df, test_df], ignore_index=True)
    X_pool = pooled_df[FEATURE_COLS].values
    y_pool = pooled_df[TARGET_COL].values
    run_ratio_sensitivity(X_pool, y_pool)