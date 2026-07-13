import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import INPUT_LEN, TARGET


def build_windows(data, split_idx: int, horizon: int):
    feature_cols = [c for c in data.columns if c != "date"]
    values = data[feature_cols].to_numpy(dtype=np.float32)
    target_index = feature_cols.index(TARGET)

    x_scaler = StandardScaler().fit(values[:split_idx])
    y_scaler = StandardScaler().fit(values[:split_idx, [target_index]])
    scaled_x = x_scaler.transform(values).astype(np.float32)
    scaled_y = y_scaler.transform(values[:, [target_index]]).ravel().astype(np.float32)

    train_x, train_y = [], []
    for start in range(0, split_idx - INPUT_LEN - horizon + 1):
        train_x.append(scaled_x[start : start + INPUT_LEN])
        train_y.append(scaled_y[start + INPUT_LEN : start + INPUT_LEN + horizon])

    test_x, test_y, test_dates = [], [], []
    first_test_start = max(0, split_idx - INPUT_LEN)
    last_start = len(data) - INPUT_LEN - horizon
    for start in range(first_test_start, last_start + 1):
        if start + INPUT_LEN < split_idx:
            continue
        test_x.append(scaled_x[start : start + INPUT_LEN])
        test_y.append(scaled_y[start + INPUT_LEN : start + INPUT_LEN + horizon])
        test_dates.append(data["date"].iloc[start + INPUT_LEN : start + INPUT_LEN + horizon].tolist())

    if not train_x or not test_x:
        raise ValueError(f"Not enough data for input={INPUT_LEN}, horizon={horizon}, split_idx={split_idx}.")

    return (
        np.array(train_x, dtype=np.float32),
        np.array(train_y, dtype=np.float32),
        np.array(test_x, dtype=np.float32),
        np.array(test_y, dtype=np.float32),
        y_scaler,
        feature_cols,
        test_dates,
    )

