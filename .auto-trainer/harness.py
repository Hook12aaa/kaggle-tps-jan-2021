"""Shared training/eval harness for all experiment worktrees.

Every variant imports this module so all experiments share one fixed
train/validation split (same seed, same row order). That invariant is what
lets the Caruana ensemble align each model's val_predictions.npy row-for-row
against a single val_labels.npy. Per-variant choices live in each worktree's
config.py; this file holds only split/fit/score logic common to every model.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CONT_COLS = [f"cont{i}" for i in range(1, 15)]
VAL_SIZE = 0.2
SPLIT_SEED = 42


def _load_features_module(features_path):
    spec = importlib.util.spec_from_file_location("features_mod", features_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_split(config):
    train_df = pd.read_csv(config["train_path"])
    test_df = pd.read_csv(config["test_path"])
    target = config["target_column"]
    id_col = config["id_column"]

    if config.get("use_features", True) and config.get("features_path"):
        fmod = _load_features_module(config["features_path"])
        train_df = fmod.engineer_features(train_df)
        test_df = fmod.engineer_features(test_df)

    drop = [c for c in (id_col, target) if c in train_df.columns]
    feature_names = [c for c in train_df.columns if c not in drop]

    X = train_df[feature_names].to_numpy(dtype=np.float64)
    y = train_df[target].to_numpy(dtype=np.float64)
    X_test = test_df[feature_names].to_numpy(dtype=np.float64)
    test_ids = test_df[id_col].to_numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=SPLIT_SEED, shuffle=True
    )
    return X_tr, X_val, y_tr, y_val, X_test, test_ids, feature_names


def build_model(config):
    mt = config["model_type"]
    hp = config.get("hyperparameters", {})
    seed = config.get("seed", 42)

    if mt == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**hp)
    if mt == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(**hp)
    if mt == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(**hp)
    if mt == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=-1, **hp)
    if mt == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(random_state=seed, **hp)
    raise ValueError(f"Unknown model_type: {mt}")


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def standardize(X_tr, X_val, X_test):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_tr - mu) / sd, (X_val - mu) / sd, (X_test - mu) / sd


def train_and_curve(config, X_tr, y_tr, X_val, y_val):
    """Fit the final model once and derive a non-degenerate validation-RMSE
    trajectory for forensics: boosting iterations for LightGBM, the loss curve
    for MLP, increasing training-data fractions for one-shot linear models.
    Inputs are already scaled by the caller when the model needs it, so the
    pickled model and the curve are produced on identical data."""
    mt = config["model_type"]
    model = build_model(config)
    model.fit(X_tr, y_tr)
    if mt == "lightgbm":
        booster = model.booster_
        n = booster.num_trees()
        pts = sorted(set(int(n * f) for f in (0.1, 0.25, 0.5, 0.75, 1.0) if int(n * f) >= 1))
        curve = [rmse(booster.predict(X_val, num_iteration=k), y_val) for k in pts]
        return model, curve
    if mt == "mlp":
        loss = getattr(model, "loss_curve_", None)
        if loss and len(loss) >= 2:
            idx = np.linspace(0, len(loss) - 1, min(8, len(loss))).astype(int)
            return model, [float(np.sqrt(loss[i])) for i in idx]
        return model, [rmse(model.predict(X_val), y_val)] * 4
    out = []
    rng = np.random.RandomState(config.get("seed", 42))
    perm = rng.permutation(len(X_tr))
    for f in (0.25, 0.5, 0.75, 1.0):
        k = max(2, int(len(X_tr) * f))
        m = build_model(config)
        m.fit(X_tr[perm[:k]], y_tr[perm[:k]])
        out.append(rmse(m.predict(X_val), y_val))
    return model, out


def count_params(model, config, n_features):
    mt = config["model_type"]
    if mt in ("ridge", "lasso", "elasticnet"):
        return int(n_features + 1)
    if mt == "lightgbm":
        hp = config.get("hyperparameters", {})
        n_est = hp.get("n_estimators", 100)
        num_leaves = hp.get("num_leaves", 31)
        return int(n_est * num_leaves)
    if mt == "mlp":
        hidden = config.get("hyperparameters", {}).get("hidden_layer_sizes", (100,))
        if isinstance(hidden, int):
            hidden = (hidden,)
        dims = [n_features] + list(hidden) + [1]
        total = 0
        for a, b in zip(dims[:-1], dims[1:]):
            total += a * b + b
        return int(total)
    raise ValueError(f"params unknown for {mt}")


def needs_scaling(config):
    return config["model_type"] in ("ridge", "lasso", "elasticnet", "mlp")
