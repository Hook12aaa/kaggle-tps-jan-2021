"""Defense-in-depth 4-layer evaluation, executed (never eyeballed).

argv: <worktree> <node_sha> <baseline_rmse> <baseline_train_rmse>
      <sig_threshold> <overfit_threshold>
Reads the worktree's metrics.json + curve.json + val arrays, computes the four
layers as numbers, writes EVALUATION.json, prints the verdict line.
metric is rmse (lower better) for this competition.
"""
import json
import math
import sys

import numpy as np


def main():
    (wt, node_sha, b_rmse, b_train, sig_thr, overfit_thr) = (
        sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]),
        float(sys.argv[5]), float(sys.argv[6]))

    m = json.load(open(f"{wt}/metrics.json"))
    curve = json.load(open(f"{wt}/curve.json"))["val_rmse_curve"]
    val_pred = np.load(f"{wt}/val_predictions.npy")

    val_rmse = m["rmse"]
    train_rmse = m["train_rmse"]

    # ---- Layer 1: data validation ----
    needed = ["rmse", "mae", "r2", "train_rmse"]
    l1_missing = [k for k in needed if k not in m]
    l1_nonfinite = [k for k in needed if k in m and not math.isfinite(m[k])]
    l1_pass = (not l1_missing) and (not l1_nonfinite) and len(val_pred) > 0 and np.isfinite(val_pred).all()
    l1 = {"passed": bool(l1_pass), "missing": l1_missing, "nonfinite": l1_nonfinite,
          "n_val": int(len(val_pred)), "epochs": len(curve)}

    # ---- Layer 2: overfitting ----
    gap = (val_rmse - train_rmse) / train_rmse
    baseline_gap = (b_rmse - b_train) / b_train
    gap_thr = max(overfit_thr, 2.0 * baseline_gap)
    tail = curve[max(0, int(0.75 * len(curve))):]
    mono_increasing = len(tail) >= 2 and all(tail[i + 1] > tail[i] for i in range(len(tail) - 1))
    l2_pass = (gap <= gap_thr) and (not mono_increasing)
    l2 = {"passed": bool(l2_pass), "gap_ratio": gap, "baseline_gap_ratio": baseline_gap,
          "gap_threshold": gap_thr, "val_curve_monotonic_increasing": bool(mono_increasing)}

    # ---- Layer 3: statistical significance (relative RMSE reduction vs baseline) ----
    rel_change = (b_rmse - val_rmse) / b_rmse
    significant = rel_change >= sig_thr
    l3 = {"passed": bool(significant), "relative_change": rel_change, "threshold": sig_thr}

    # ---- Layer 4: training forensics ----
    nan_epochs = [i for i, v in enumerate(curve) if not math.isfinite(v)]
    diffs = [abs(curve[i + 1] - curve[i]) for i in range(len(curve) - 1)]
    spikes = [i for i, d in enumerate(diffs) if d > 0.05 * abs(curve[i])]
    mode_collapse = float(np.std(val_pred)) < 1e-6
    l4_pass = (not nan_epochs) and (not spikes) and (not mode_collapse)
    l4 = {"passed": bool(l4_pass), "nan_epochs": nan_epochs, "spikes": spikes,
          "mode_collapse": bool(mode_collapse), "val_pred_std": float(np.std(val_pred))}

    if not l1["passed"]:
        verdict = "REJECT"; reason = "layer1 data validation"
    elif not l2["passed"]:
        verdict = "REJECT"; reason = "layer2 overfitting"
    elif not l4["passed"]:
        verdict = "REJECT"; reason = "layer4 forensics"
    elif l3["passed"]:
        verdict = "ACCEPT"; reason = "all layers pass, improvement significant"
    else:
        verdict = "INCONCLUSIVE"; reason = "passes validation but improvement not significant"

    out = {"node_sha": node_sha, "verdict": verdict, "reason": reason,
           "primary_metric": {"name": "rmse", "value": val_rmse, "baseline_value": b_rmse,
                              "relative_change": rel_change},
           "secondary_metrics": [
               {"name": "mae", "value": m["mae"]},
               {"name": "r2", "value": m["r2"]}],
           "trainable_params": m["trainable_params"],
           "layers": {"data_validation": l1, "overfitting": l2,
                      "statistical_significance": l3, "forensics": l4}}
    json.dump(out, open(f"{wt}/EVALUATION.json", "w"), indent=2)
    print(json.dumps({"verdict": verdict, "reason": reason, "rmse": val_rmse,
                      "rel_change": rel_change, "gap": gap, "params": m["trainable_params"]}))


if __name__ == "__main__":
    main()
