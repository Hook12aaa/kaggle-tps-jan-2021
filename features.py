"""Feature engineering for TPS-Jan-2021 (locked by feature-engineer skill).

Single entry point: engineer_features(df) -> augmented df.
All transformations are deterministic row-wise math on the cont1..cont14 columns.
No target is referenced (leakage-free). Row count is preserved.
"""

CONT_COLS = [f"cont{i}" for i in range(1, 15)]


def engineer_features(df):
    missing = [c for c in CONT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"engineer_features: missing required source columns: {missing}")

    out = df.copy()
    cont = out[CONT_COLS]

    out["row_mean"] = cont.mean(axis=1)
    out["row_std"] = cont.std(axis=1)
    out["row_min"] = cont.min(axis=1)
    out["row_max"] = cont.max(axis=1)
    out["row_range"] = out["row_max"] - out["row_min"]
    out["row_median"] = cont.median(axis=1)
    out["row_sumsq"] = (cont ** 2).sum(axis=1)

    out["cont7_x_cont2"] = out["cont7"] * out["cont2"]
    out["cont7_x_cont3"] = out["cont7"] * out["cont3"]

    out["cont11_x_cont12"] = out["cont11"] * out["cont12"]
    out["cont11_minus_cont12"] = out["cont11"] - out["cont12"]
    out["cont11_div_cont12"] = out["cont11"] / (out["cont12"] + 1.5)
    out["cont11_plus_cont12"] = out["cont11"] + out["cont12"]
    out["cont11_cont12_mean"] = (out["cont11"] + out["cont12"]) / 2.0

    out["cont1_x_cont3"] = out["cont1"] * out["cont3"]
    out["cont2_x_cont6"] = out["cont2"] * out["cont6"]
    out["cont9_sq"] = out["cont9"] ** 2

    return out


ENGINEERED_COLUMNS = [
    "row_mean", "row_std", "row_min", "row_max", "row_range", "row_median",
    "row_sumsq", "cont7_x_cont2", "cont7_x_cont3", "cont11_x_cont12",
    "cont11_minus_cont12", "cont11_div_cont12", "cont11_plus_cont12",
    "cont11_cont12_mean", "cont1_x_cont3", "cont2_x_cont6", "cont9_sq",
]
