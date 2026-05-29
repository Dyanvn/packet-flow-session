import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    classification_report,

    confusion_matrix,

    roc_curve,

    auc
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


X = df[selected_features]

y = df['Label']


# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# =========================
# LOAD MODEL
# =========================

model = joblib.load(

    "outputs/models/xgboost_flow_model.pkl"
)


# =========================
# PREDICT
# =========================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# REPORT
# =========================

print(

    classification_report(

        y_test,

        y_pred,

        digits=6
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(

    y_test,

    y_pred
)


plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(

    "outputs/figures/confusion_matrix.png"
)

plt.show()


# =========================
# ROC CURVE
# =========================

fpr, tpr, _ = roc_curve(

    y_test,

    y_prob
)

roc_auc = auc(

    fpr,

    tpr
)


plt.figure(figsize=(6,5))

plt.plot(

    fpr,

    tpr,

    label=f"AUC = {roc_auc:.6f}"
)

plt.plot(

    [0,1],

    [0,1],

    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()


plt.savefig(

    "outputs/figures/roc_curve.png"
)

plt.show()