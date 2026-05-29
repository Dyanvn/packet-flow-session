import pandas as pd


df = pd.read_csv(

    "data/processed/full_clean_dataset.csv"
)


# =========================
# LABEL DISTRIBUTION
# =========================

label_counts = df['Label'].value_counts()

label_percent = df['Label'].value_counts(normalize=True) * 100


print("\n========== LABEL COUNTS ==========")

print(label_counts)


print("\n========== LABEL PERCENT ==========")

print(label_percent)