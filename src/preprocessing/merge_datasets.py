import pandas as pd


# =========================
# DATASET FILES
# =========================

dataset_files = [

    "Tuesday-WorkingHours.pcap_ISCX.csv",

    "Wednesday-workingHours.pcap_ISCX.csv",

    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",

    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]


# =========================
# LOAD DATASETS
# =========================

dfs = []

for file in dataset_files:

    print(f"[INFO] Loading: {file}")

    df = pd.read_csv(

        f"data/raw/{file}"
    )

    dfs.append(df)


# =========================
# MERGE
# =========================

merged_df = pd.concat(

    dfs,

    ignore_index=True
)


print("\n========== MERGED DATASET ==========")

print(merged_df.shape)


# =========================
# SAVE
# =========================

merged_df.to_csv(

    "data/processed/merged_dataset.csv",

    index=False
)


print("\n[INFO] Dataset merged successfully")