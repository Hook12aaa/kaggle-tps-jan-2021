import csv
import json
import pickle
import sys

import numpy as np
sys.path.insert(0, "/Users/hook/Documents/coding/python/kaggle/tps-jan-2021/.auto-trainer")
import harness
from config import CONFIG


def main():
    X_tr, X_val, y_tr, y_val, X_test, ids, fnames = harness.load_split(CONFIG)
    if harness.needs_scaling(CONFIG):
        X_tr, X_val, X_test = harness.standardize(X_tr, X_val, X_test)
    model = pickle.load(open("model.pkl", "rb"))["model"]
    val_pred = np.asarray(model.predict(X_val), dtype=float)
    tr_pred = np.asarray(model.predict(X_tr), dtype=float)
    test_pred = np.asarray(model.predict(X_test), dtype=float)

    val_rmse = harness.rmse(val_pred, y_val)
    train_rmse = harness.rmse(tr_pred, y_tr)
    mae = float(np.mean(np.abs(val_pred - y_val)))
    ss_res = float(np.sum((y_val - val_pred) ** 2))
    ss_tot = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    params = harness.count_params(model, CONFIG, len(fnames))

    np.save("val_predictions.npy", val_pred)
    np.save("val_labels.npy", np.asarray(y_val, dtype=float))

    metrics = {"rmse": val_rmse, "train_rmse": train_rmse, "mae": mae, "r2": r2,
               "trainable_params": params, "val_pred_std": float(np.std(val_pred))}
    json.dump(metrics, open("metrics.json", "w"))

    with open("submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([CONFIG["id_column"], CONFIG["prediction_column"]])
        for i, p in zip(ids, test_pred):
            w.writerow([int(i), float(p)])

    print(json.dumps(metrics))
    print("eval OK")


if __name__ == "__main__":
    main()
