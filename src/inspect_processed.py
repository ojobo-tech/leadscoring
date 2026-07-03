from pathlib import Path

import pandas as pd


PROCESSED_FILE = Path("data/processed/lead_master.csv")


def main() -> None:
    if not PROCESSED_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PROCESSED_FILE}")

    df = pd.read_csv(PROCESSED_FILE)

    print("=" * 80)
    print("PROCESSED DATA INSPECTION")
    print("=" * 80)
    print(f"Shape: {df.shape}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values (top 20):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nTarget balance:")
    if "lead_is_converted" in df.columns:
        print(df["lead_is_converted"].value_counts(dropna=False))
    else:
        print("lead_is_converted column not found.")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()