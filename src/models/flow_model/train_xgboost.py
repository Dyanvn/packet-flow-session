import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.preprocessing.feature_selection import (

    selected_features
)


# =========================
# LOAD DATASET
# =========================

print("[INFO] Loading dataset...")


df = pd.read_csv(

    "data/processed/full_clean_dataset.csv"
)


# =========================
# FEATURES
# =========================

X = df[selected_features]

y = df['Label']


# =========================
# SPLIT DATASET
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\n========== DATA SHAPES ==========")

print("X_train:", X_train.shape)

print("X_test :", X_test.shape)


# =========================
# CLASS WEIGHT
# =========================

attack_count = y_train.sum()

benign_count = len(y_train) - attack_count

scale_pos_weight = benign_count / attack_count


print("\n========== CLASS INFO ==========")

print("Benign :", benign_count)

print("Attack :", attack_count)

print("Scale Pos Weight:", scale_pos_weight)


# =========================
# BUILD MODEL
# =========================

model = XGBClassifier(

    n_estimators=100,

    max_depth=4,

    learning_rate=0.1,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    eval_metric='logloss',

    random_state=42
)


# =========================
# TRAIN MODEL
# =========================

print("\n[INFO] Training XGBoost model...")


model.fit(

    X_train,

    y_train
)


print("[INFO] Training completed")


# =========================
# PREDICT
# =========================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# METRICS
# =========================

accuracy = accuracy_score(

    y_test,

    y_pred
)

precision = precision_score(

    y_test,

    y_pred
)

recall = recall_score(

    y_test,

    y_pred
)

f1 = f1_score(

    y_test,

    y_pred
)

auc = roc_auc_score(

    y_test,

    y_prob
)


print("\n========== MODEL METRICS ==========")

print(f"Accuracy : {accuracy:.6f}")

print(f"Precision: {precision:.6f}")

print(f"Recall   : {recall:.6f}")

print(f"F1 Score : {f1:.6f}")

print(f"ROC-AUC  : {auc:.6f}")


# =========================
# SAVE MODEL
# =========================

joblib.dump(

    model,

    "outputs/models/xgboost_flow_model.pkl"
)


print("\n[INFO] Model saved successfully")