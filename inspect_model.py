import joblib

models = {
    "PACKET": "outputs/models/xgboost_packet_model.pkl",
    "FLOW": "outputs/models/xgboost_flow_model.pkl",
    "SESSION": "outputs/models/xgboost_session_model.pkl",
}

for name, path in models.items():
    print("\n" + "="*70)
    print(f"{name} MODEL")
    print("="*70)

    model = joblib.load(path)

    print("Model Type:", type(model))

    if hasattr(model, "n_features_in_"):
        print("Features:", model.n_features_in_)

    if hasattr(model, "classes_"):
        print("Classes:", model.classes_)

    print("\nParameters:")
    for k, v in model.get_params().items():
        print(f"{k}: {v}")