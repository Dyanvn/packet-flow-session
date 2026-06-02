import pandas as pd
import numpy as np


def validate_dataframe(df, name):
    print(f"\n{'='*60}")
    print(f"Validation Report: {name}")
    print(f"{'='*60}")
    
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values")
    else:
        print(missing[missing > 0].to_string())
    
    print(f"\nInfinite values:")
    inf_mask = df.select_dtypes(include=[np.number]).isin([np.inf, -np.inf]).sum()
    if inf_mask.sum() == 0:
        print("  No infinite values")
    else:
        print(inf_mask[inf_mask > 0].to_string())
    
    print(f"\nData types:")
    print(df.dtypes.to_string())
    
    print(f"\nBasic statistics:")
    print(df.describe().to_string())
    
    print(f"\nColumns: {df.columns.tolist()}")


def main():
    print("[TRUNG] Validating output features...")
    
    try:
        packet_df = pd.read_csv("outputs/features/packet_features.csv")
        validate_dataframe(packet_df, "Packet Features")
    except Exception as e:
        print(f"Error reading packet features: {e}")
    
    try:
        flow_df = pd.read_csv("outputs/features/flow_features.csv")
        validate_dataframe(flow_df, "Flow Features")
    except Exception as e:
        print(f"Error reading flow features: {e}")
    
    print(f"\n{'='*60}")
    print("Validation completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
