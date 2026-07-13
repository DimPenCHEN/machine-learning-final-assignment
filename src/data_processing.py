import random
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

from .config import DATA_DIR, HORIZONS, MEAN_COLUMNS, POWER_COLUMNS, ROOT, TARGET, UCI_URL, WEATHER_COLUMNS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        renamed[col] = key
    df = df.rename(columns=renamed)
    if "date" not in df.columns and "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
    return df


def download_uci() -> str:
    DATA_DIR.mkdir(exist_ok=True)
    txt_path = DATA_DIR / "household_power_consumption.txt"
    if txt_path.exists():
        return str(txt_path)

    zip_path = DATA_DIR / "household_power_consumption.zip"
    print(f"Downloading UCI data to {zip_path} ...")
    urllib.request.urlretrieve(UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("household_power_consumption.txt", DATA_DIR)
    return str(txt_path)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    if "date" not in df.columns:
        raise ValueError("Input data must contain Date/date or datetime column.")

    if "time" in df.columns:
        dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), dayfirst=True, errors="coerce")
    else:
        dt = pd.to_datetime(df["date"], errors="coerce")
    df = df.assign(date=dt.dt.floor("D")).dropna(subset=["date"])

    numeric_cols = [c for c in POWER_COLUMNS + MEAN_COLUMNS + WEATHER_COLUMNS if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3"}.issubset(df.columns):
        df["sub_metering_remainder"] = (
            df["global_active_power"] * 1000.0 / 60.0
            - (df["sub_metering_1"] + df["sub_metering_2"] + df["sub_metering_3"])
        )

    agg = {}
    for col in POWER_COLUMNS + ["sub_metering_remainder"]:
        if col in df.columns:
            agg[col] = "sum"
    for col in MEAN_COLUMNS:
        if col in df.columns:
            agg[col] = "mean"
    for col in WEATHER_COLUMNS:
        if col in df.columns:
            agg[col] = "first"

    daily = df.groupby("date", as_index=False).agg(agg).sort_values("date").reset_index(drop=True)
    daily = daily.interpolate(limit_direction="both").ffill().bfill()
    if TARGET not in daily.columns:
        raise ValueError(f"Target column {TARGET!r} is missing after aggregation.")
    return daily


def read_raw_or_provided() -> tuple[pd.DataFrame, int, str]:
    train_path = next((ROOT / name for name in ("train.csv", "Train.csv") if (ROOT / name).exists()), None)
    test_path = next(
        (ROOT / name for name in ("test.csv", "tes.csv", "Test.csv", "Tes.csv") if (ROOT / name).exists()),
        None,
    )
    if train_path and test_path:
        train_daily = aggregate_daily(pd.read_csv(train_path))
        test_daily = aggregate_daily(pd.read_csv(test_path))
        data = pd.concat([train_daily, test_daily], ignore_index=True)
        return data, len(train_daily), f"provided split: {train_path.name} + {test_path.name}"

    txt_path = download_uci()
    raw = pd.read_csv(txt_path, sep=";", na_values="?", low_memory=False)
    daily = aggregate_daily(raw)
    split_idx = len(daily) - max(HORIZONS) - 1
    split_idx = max(split_idx, int(len(daily) * 0.7))
    split_idx = min(split_idx, len(daily) - max(HORIZONS) - 1)
    return daily, split_idx, "UCI public data chronological split"

