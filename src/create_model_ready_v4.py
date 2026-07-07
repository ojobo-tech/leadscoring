from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_master_v3.csv"

OUTPUT_WITH_OWNER = PROCESSED_DIR / "lead_model_ready_v4_with_owner.csv"
OUTPUT_WITHOUT_OWNER = PROCESSED_DIR / "lead_model_ready_v4_no_owner.csv"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)


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


def build_model_ready(df: pd.DataFrame, drop_lead_owner_id: bool) -> pd.DataFrame:
    df = df.copy()

    # Convert booleans to numeric
    df = standardize_booleans_to_int(
        df,
        [
            "lead_is_converted",
            "owner_is_active",
        ],
    )

    # Drop constant / leakage-prone / raw helper columns
    drop_cols = [
        "lead_id",
        "lead_created_date",
        "owner_hire_date",
        "last_task_created_date",
        "last_task_activity_date",
        "converted_account_id",
        "converted_contact_id",
        "converted_opportunity_id",
        "owner_is_active",   # constant in your EDA
    ]

    if drop_lead_owner_id:
        drop_cols.append("lead_owner_id")

    model_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Put target at the end
    target_col = "lead_is_converted"
    cols = [c for c in model_df.columns if c != target_col] + [target_col]
    model_df = model_df[cols]

    return model_df


def main() -> None:
    df = load_data()

    with_owner = build_model_ready(df, drop_lead_owner_id=False)
    no_owner = build_model_ready(df, drop_lead_owner_id=True)

    with_owner.to_csv(OUTPUT_WITH_OWNER, index=False)
    no_owner.to_csv(OUTPUT_WITHOUT_OWNER, index=False)

    print(f"Saved: {OUTPUT_WITH_OWNER}  Shape: {with_owner.shape}")
    print(f"Saved: {OUTPUT_WITHOUT_OWNER}  Shape: {no_owner.shape}")

    print("\nWITH OWNER columns:")
    print(with_owner.columns.tolist())

    print("\nWITHOUT OWNER columns:")
    print(no_owner.columns.tolist())


if __name__ == "__main__":
    main()