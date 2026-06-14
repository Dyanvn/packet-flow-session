# ui/realtime_dashboard.py

from pathlib import Path
import time

import pandas as pd
import streamlit as st
from datetime import datetime


LOG_PATH = Path("outputs/metrics/realtime_alerts_log.csv")

st.set_page_config(
    page_title="Multi-layer IDS Realtime Dashboard",
    layout="wide",
)

st.title(
    "🔥 Hệ thống phát hiện đe doạ đa lớp "
    "(Packet - Flow - Session) - Realtime"
)

placeholder = st.empty()

REFRESH_SEC = 2


def load_alerts():
    if not LOG_PATH.exists():
        return pd.DataFrame(
            columns=[
                "log_time",
                "packet_time",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "protocol",
                "packet_proba",
                "flow_proba",
                "session_proba",
                "ensemble_pred",
            ]
        )

    try:
        df = pd.read_csv(LOG_PATH)

        if "log_time" in df.columns:
            df = df.sort_values(
                "log_time",
                ascending=False
            )

        return df

    except Exception as e:
        st.error(f"Lỗi đọc file log: {e}")
        return pd.DataFrame()


while True:

    df = load_alerts()

    with placeholder.container():

        st.subheader("📡 Realtime Alert Monitoring")

        if df.empty:

            st.info(
                "Chưa có alert nào. "
                "Hãy chạy demo_realtime_console.py phía backend."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Tổng Alert",
                    len(df)
                )

            with col2:
                st.metric(
                    "Alert đang hiển thị",
                    min(len(df), 20)
                )

            st.subheader("🛑 Danh sách Alert Mới Nhất")

            df_view = df.head(20).copy()

            def make_desc(row):

                # ===== Packet Time =====
                packet_time_display = str(row.get("packet_time", ""))

                try:
                    ts = float(row["packet_time"])

                    # chỉ convert nếu là epoch hợp lệ
                    if ts > 1000000000:
                        packet_time_display = datetime.fromtimestamp(
                            ts
                        ).strftime("%Y-%m-%d %H:%M:%S")

                except Exception:
                    pass

                # ===== Detect Attack Type =====
                attack_type = "Bất thường mạng"

                try:
                    dst_port = int(row["dst_port"])
                except Exception:
                    dst_port = -1

                protocol = str(row.get("protocol", "")).upper()

                if protocol == "TCP" and dst_port in (80, 443):
                    attack_type = "Nghi ngờ DDoS HTTP / SYN Flood"

                elif protocol == "TCP" and dst_port == 22:
                    attack_type = "Nghi ngờ brute-force SSH / scan"

                else:
                    attack_type = "Nghi ngờ port scan / traffic bất thường"

                return (
                    "[ALERT]\n\n"
                    f"Thời gian tấn công (Packet) : {packet_time_display}\n"
                    f"Thời gian phát hiện (AI)   : {row['log_time']}\n\n"
                    f"Nguồn tấn công      : {row['src_ip']}:{row['src_port']}\n"
                    f"Máy mục tiêu        : {row['dst_ip']}:{row['dst_port']}\n"
                    f"Giao thức           : {protocol}\n"
                    f"Loại nghi ngờ       : {attack_type}\n\n"
                    "Độ tin cậy AI\n"
                    f"  Packet Model  : {float(row['packet_proba']):.6f}\n"
                    f"  Flow Model    : {float(row['flow_proba']):.6f}\n"
                    f"  Session Model : {float(row['session_proba']):.6f}"
                )

            df_view["description"] = df_view.apply(
                make_desc,
                axis=1
            )

            for _, row in df_view.iterrows():
                st.markdown("```text\n" + row["description"] + "\n```")
                st.markdown("---")

            if "log_time" in df.columns:

                try:

                    df_time = df.copy()
                    df_time["packet_time"] = pd.to_datetime(df_time["packet_time"], unit="s")
                    df_time.set_index("packet_time", inplace=True)
                    df_count = df_time.resample("1s")["ensemble_pred"].count()

                    st.subheader(
                        "📈 Số lượng Alert theo thời gian"
                    )

                    st.line_chart(
                        df_count,
                        height=250
                    )

                except Exception as e:

                    st.warning(
                        f"Không thể vẽ biểu đồ: {e}"
                    )

    time.sleep(REFRESH_SEC)
