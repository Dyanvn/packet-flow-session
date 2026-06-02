import os
import sys
from glob import glob

import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing.feature_selection import selected_features

RAW_DIR = "data/raw"
MERGED_PATH = "data/processed/merged_dataset.csv"
CLEAN_PATH = "data/processed/full_clean_dataset.csv"
MODEL_DATA_PATH = "data/processed/flow_model_dataset.csv"


def ensure_output_dirs():
    os.makedirs(os.path.dirname(MERGED_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)


def load_raw_files(raw_dir):
    paths = sorted(glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No raw CSV files found in {raw_dir}")
    return paths


def read_and_clean_raw(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def merge_datasets(paths):
    frames = []
    for path in paths:
        print(f"[INFO] Loading raw file: {path}")
        frames.append(read_and_clean_raw(path))
    merged = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Merged dataset shape: {merged.shape}")
    merged.to_csv(MERGED_PATH, index=False)
    print(f"[INFO] Saved merged dataset: {MERGED_PATH}")
    return merged


def clean_dataset(df):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    if "Label" not in df.columns:
        raise KeyError("Label column not found in raw dataset")
    df["Label"] = df["Label"].astype(str).str.strip().apply(lambda x: 0 if x.upper() == "BENIGN" else 1)
    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[INFO] Saved clean dataset: {CLEAN_PATH}")
    return df


def build_model_dataset(df):
    missing = [feat for feat in selected_features if feat not in df.columns]
    if missing:
        raise ValueError(f"Selected features missing from dataset: {missing}")
    model_df = df[selected_features + ["Label"]].copy()
    model_df.to_csv(MODEL_DATA_PATH, index=False)
    print(f"[INFO] Saved model-ready dataset: {MODEL_DATA_PATH}")
    return model_df


def summarize(df, name):
    print(f"\n=== Summary: {name} ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(df.dtypes.to_string())
    if df.select_dtypes(include=["number"]).shape[1] > 0:
        print(df.describe().transpose().loc[:, ["count", "mean", "std", "min", "max"]])


def main():
    ensure_output_dirs()
    raw_paths = load_raw_files(RAW_DIR)
    merged = merge_datasets(raw_paths)
    cleaned = clean_dataset(merged)
    model_df = build_model_dataset(cleaned)
    summarize(model_df, "Flow Model Dataset")
    print("\n[INFO] Flow dataset build complete.")


if __name__ == "__main__":
    main()
