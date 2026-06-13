# src/models/session_model/train_xgboost.py

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def load_session_dataset(session_csv_path: str | Path) -> pd.DataFrame:
    session_csv_path = Path(session_csv_path)
    df = pd.read_csv(session_csv_path)

    # Tạo nhãn session: có ít nhất 1 flow attack → label = 1
    df["SessionLabel"] = (df["num_attack_flows"] > 0).astype(int)

    return df

def split_features_labels(df: pd.DataFrame):
    """
    Chọn các cột feature numeric cho model.
    Bạn có thể chỉnh lại danh sách feature cho phù hợp.
    """
    feature_cols = [
        "avg_flow_duration",
        "max_flow_duration",
        "total_fwd_packets",
        "avg_fwd_packets",
        "total_bwd_packets",
        "avg_bwd_packets",
        "avg_packet_size",
        "avg_packet_length",
        "num_flows",
    ]

    X = df[feature_cols]
    y = df["SessionLabel"]
    return X, y

def train_session_xgboost(
    session_csv_path: str | Path = "data/processed/session_model_dataset.csv",
    model_output_path: str | Path = "outputs/models/xgboost_session_model.pkl",
    metrics_output_dir: str | Path = "outputs/metrics",
    test_size: float = 0.2,
    random_state: int = 42,
):
    session_csv_path = Path(session_csv_path)
    model_output_path = Path(model_output_path)
    metrics_output_dir = Path(metrics_output_dir)

    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_session_dataset(session_csv_path)
    X, y = split_features_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Đánh giá nhanh
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    # Lưu metrics ra CSV cho đồng bộ với flow/packet
    report_df = pd.DataFrame(report).transpose()
    report_path = metrics_output_dir / "session_classification_report.csv"
    report_df.to_csv(report_path)

    auc_path = metrics_output_dir / "session_auc.txt"
    with auc_path.open("w", encoding="utf-8") as f:
        f.write(f"AUC: {auc}\n")

    print(f"[SessionModel] AUC: {auc:.4f}")
    print(f"[SessionModel] Classification report saved to: {report_path}")

    # Lưu model
    import joblib

    joblib.dump(model, model_output_path)
    print(f"[SessionModel] Model saved to: {model_output_path}")

if __name__ == "__main__":
    train_session_xgboost()