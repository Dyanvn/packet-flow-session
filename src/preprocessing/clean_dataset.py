import pandas as pd
import numpy as np


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(

    "data/processed/merged_dataset.csv"
)


print("[INFO] Original Shape:", df.shape)


# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = df.columns.str.strip()


# =========================
# REMOVE INFINITY
# =========================

df.replace(

    [np.inf, -np.inf],

    np.nan,

    inplace=True
)


# =========================
# DROP NaN
# =========================

df.dropna(inplace=True)


# =========================
# LABEL ENCODING
# =========================

df['Label'] = df['Label'].apply(

    lambda x: 0 if x == 'BENIGN' else 1
)


print("\n========== CLEANED DATASET ==========")

print(df.shape)


# =========================
# SAVE
# =========================

df.to_csv(

    "data/processed/full_clean_dataset.csv",

    index=False
)


print("\n[INFO] Clean dataset saved successfully")