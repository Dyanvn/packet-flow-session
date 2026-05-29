import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocessing.feature_selection import (

    selected_features
)


# =========================
# LOAD MODEL
# =========================

model = joblib.load(

    "outputs/models/xgboost_flow_model.pkl"
)


# =========================
# IMPORTANCE
# =========================

importance = model.feature_importances_


importance_df = pd.DataFrame({

    'Feature': selected_features,

    'Importance': importance
})


importance_df = importance_df.sort_values(

    by='Importance',

    ascending=False
)


print(importance_df)


# =========================
# PLOT
# =========================

plt.figure(figsize=(10,6))

plt.bar(

    importance_df['Feature'],

    importance_df['Importance']
)

plt.xticks(rotation=45)

plt.title("Feature Importance")

plt.tight_layout()


plt.savefig(

    "outputs/figures/feature_importance.png"
)

plt.show()