import joblib
import pandas as pd


# =========================
# LOAD MODEL
# =========================

model = joblib.load(

    "outputs/models/xgboost_flow_model.pkl"
)


# =========================
# SAMPLE FLOW
# =========================

sample_flow = {

    'Flow Duration': 500,

    'Total Fwd Packets': 20,

    'Total Backward Packets': 5,

    'SYN Flag Count': 2,

    'ACK Flag Count': 18,

    'Average Packet Size': 450,

    'Packet Length Mean': 420
}


sample_df = pd.DataFrame(

    [sample_flow]
)


# =========================
# PREDICT
# =========================

prediction = model.predict(

    sample_df
)[0]


probability = model.predict_proba(

    sample_df
)[0][1]


# =========================
# RESULT
# =========================

if prediction == 0:

    print("\nPrediction: BENIGN")

else:

    print("\nPrediction: ATTACK")


print(f"Attack Probability: {probability:.6f}")