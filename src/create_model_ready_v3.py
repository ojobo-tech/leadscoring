from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_master_v3.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_model_ready_v3.csv"


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


def build_model_ready_v3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert booleans to numeric
    df = standardize_booleans_to_int(
        df,
        [
            "lead_is_converted",
            "owner_is_active",
        ],
    )

    # Parse date fields so they can be safely removed after recency features were created upstream
    df = parse_dates(
        df,
        [
            "lead_created_date",
            "owner_hire_date",
            "last_task_created_date",
            "last_task_activity_date",
        ],
    )

    # Drop leakage-prone, raw date, and direct identifier fields
    drop_cols = [
        "lead_id",
        "lead_created_date",
        "owner_hire_date",
        "last_task_created_date",
        "last_task_activity_date",
        "converted_account_id",
        "converted_contact_id",
        "converted_opportunity_id",
    ]

    model_df = df.drop(columns=drop_cols, errors="ignore")

    # Put target at the end
    target_col = "lead_is_converted"
    cols = [c for c in model_df.columns if c != target_col] + [target_col]
    model_df = model_df[cols]

    return model_df


def main() -> None:
    df = load_data()
    model_df = build_model_ready_v3(df)

    model_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved model-ready dataset to: {OUTPUT_FILE}")
    print(f"Shape: {model_df.shape}")
    print("Columns:")
    print(model_df.columns.tolist())


if __name__ == "__main__":
    main()