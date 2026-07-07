from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_master.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_model_ready.csv"


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Convert dates for recency features
    if "lead_created_date" in df.columns:
        df["lead_created_date"] = to_datetime(df["lead_created_date"])

    if "last_task_created_date" in df.columns:
        df["last_task_created_date"] = to_datetime(df["last_task_created_date"])

    if "last_task_activity_date" in df.columns:
        df["last_task_activity_date"] = to_datetime(df["last_task_activity_date"])

    if "owner_hire_date" in df.columns:
        df["owner_hire_date"] = to_datetime(df["owner_hire_date"])

    # Create useful time-based features
    if "lead_created_date" in df.columns:
        reference_date = df["lead_created_date"].max()
        df["lead_age_days"] = (reference_date - df["lead_created_date"]).dt.days

    if "owner_hire_date" in df.columns and "lead_created_date" in df.columns:
        df["owner_tenure_days"] = (df["lead_created_date"] - df["owner_hire_date"]).dt.days

    if "lead_created_date" in df.columns and "last_task_created_date" in df.columns:
        df["days_since_last_task_created"] = (
            df["lead_created_date"] - df["last_task_created_date"]
        ).dt.days

    if "lead_created_date" in df.columns and "last_task_activity_date" in df.columns:
        df["days_since_last_task_activity"] = (
            df["lead_created_date"] - df["last_task_activity_date"]
        ).dt.days

    # Drop obvious leakage / direct ID / PII fields
    drop_cols = [
        "lead_id",
        "first_name",
        "last_name",
        "company",
        "email",
        "phone",
        "converted_account_id",
        "converted_contact_id",
        "converted_opportunity_id",
        "owner_name",
        "owner_title",
        "lead_created_date",
        "last_task_created_date",
        "last_task_activity_date",
        "owner_hire_date",
    ]

    model_df = df.drop(columns=drop_cols, errors="ignore")

    # Make booleans numeric where appropriate
    bool_cols = [
        "owner_is_active",
        "lead_is_converted",
    ]
    for col in bool_cols:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype("int64", errors="ignore")

    # Save output
    model_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved model-ready dataset to: {OUTPUT_FILE}")
    print(f"Shape: {model_df.shape}")
    print("Columns:")
    print(model_df.columns.tolist())


if __name__ == "__main__":
    main()