import json
import os
import sys
sys.path.insert(0, "/Users/hook/Documents/coding/python/kaggle/tps-jan-2021/.auto-trainer")
from config import CONFIG


def main():
    for key in ("train_path", "test_path", "features_path"):
        assert os.path.exists(CONFIG[key]), f"missing path: {CONFIG[key]}"
    assert os.path.exists("constraints.lock"), "missing constraints.lock"
    lock = json.load(open("constraints.lock"))
    assert lock.get("constraints_hash"), "constraints.lock has no constraints_hash"
    assert CONFIG.get("model_type"), "config missing model_type"
    print("preflight OK")


if __name__ == "__main__":
    main()
