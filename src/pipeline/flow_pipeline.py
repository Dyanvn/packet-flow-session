import joblib

import pandas as pd

from src.ingestion.packet_reader import load_packets

from src.features.flow.flow_builder import (

    build_flows
)

from src.features.flow.flow_features import (

    extract_flow_features
)


# =========================
# LOAD MODEL
# =========================

model = joblib.load(

    "outputs/models/xgboost_flow_model.pkl"
)


# =========================
# LOAD PCAP
# =========================

packets = load_packets(

    "data/sample/Friday-WorkingHours.pcap",

    limit=10000
)


# =========================
# BUILD FLOWS
# =========================

flows = build_flows(

    packets
)


print(f"[INFO] Total flows: {len(flows)}")


# =========================
# INFERENCE
# =========================

for flow_key, flow_packets in flows.items():

    features = extract_flow_features(

        flow_packets
    )

    if features is None:
        continue

    df = pd.DataFrame(

        [features]
    )

    prediction = model.predict(

        df
    )[0]

    probability = model.predict_proba(

        df
    )[0][1]

    print()

    print(flow_key)

    print(
        f"Prediction: {'ATTACK' if prediction == 1 else 'BENIGN'}"
    )

    print(
        f"Attack Probability: {probability:.4f}"
    )