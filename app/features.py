import math

import pandas as pd

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

RAW_FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
ENGINEERED_FEATURES = ["power_w", "strain", "temp_diff_k"]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Type"] = out["Type"].map(TYPE_MAP)
    out["power_w"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] * 2 * math.pi / 60
    out["strain"] = out["Torque [Nm]"] * out["Tool wear [min]"]
    out["temp_diff_k"] = out["Process temperature [K]"] - out["Air temperature [K]"]
    return out[FEATURES]
