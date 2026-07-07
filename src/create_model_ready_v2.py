from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_master_v2.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_model_ready_v2.csv"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def standardize_booleans_to_int(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": 1,
                        "false": 0,
                        "1": 1,
                        "0": 0,
                        "yes": 1,
                        "no": 0,
                    }
                )
            )
    return df


def build_model_ready_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert target and key boolean fields
    df = standardize_booleans_to_int(
        df,
        [
            "lead_is_converted",
            "owner_is_active",
            "account_is_active",
        ],
    )

    # Parse any date-like columns that may still remain
    df = parse_dates(df, ["account_created_date", "last_contact_event_date"])

    # Remove leakage-prone, direct ID, and text fields not needed for modeling
    drop_cols = [
        "lead_id",
        "company",
        "company_norm",
        "user_id",
        "account_id",
        "account_number",
        "account_created_date",
        "last_contact_event_date",
    ]

    model_df = df.drop(columns=drop_cols, errors="ignore")

    # Put target at the end
    target_col = "lead_is_converted"
    cols = [c for c in model_df.columns if c != target_col] + [target_col]
    model_df = model_df[cols]

    return model_df


def main() -> None:
    df = load_data()
    model_df = build_model_ready_v2(df)

    model_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved model-ready dataset to: {OUTPUT_FILE}")
    print(f"Shape: {model_df.shape}")
    print("Columns:")
    print(model_df.columns.tolist())


if __name__ == "__main__":
    main()