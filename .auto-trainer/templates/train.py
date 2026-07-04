import json
import pickle
import sys
sys.path.insert(0, "/Users/hook/Documents/coding/python/kaggle/tps-jan-2021/.auto-trainer")
import harness
from config import CONFIG


def main():
    X_tr, X_val, y_tr, y_val, X_test, ids, fnames = harness.load_split(CONFIG)
    if harness.needs_scaling(CONFIG):
        X_tr, X_val, X_test = harness.standardize(X_tr, X_val, X_test)
    model, curve = harness.train_and_curve(CONFIG, X_tr, y_tr, X_val, y_val)
    pickle.dump({"model": model, "fnames": fnames}, open("model.pkl", "wb"))
    for i, v in enumerate(curve):
        print(json.dumps({"epoch": i + 1, "val_rmse": v}))
    json.dump({"val_rmse_curve": curve}, open("curve.json", "w"))
    print("train OK")


if __name__ == "__main__":
    main()
