import pandas as pd

from sklearn.model_selection import (

    train_test_split
)

from src.preprocessing.feature_selection import (

    selected_features
)


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(

    "data/processed/full_clean_dataset.csv"
)


# =========================
# FEATURES
# =========================

X = df[selected_features]

y = df['Label']


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\n========== SPLIT RESULTS ==========")

print("X_train:", X_train.shape)

print("X_test :", X_test.shape)

print("y_train:", y_train.shape)

print("y_test :", y_test.shape)