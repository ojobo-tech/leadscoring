from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_model_ready_v5.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_model_ready_v6.csv"


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


def safe_clip_nonnegative(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def recompute_recency_scores(df: pd.DataFrame) -> pd.DataFrame:
    if "days_since_last_task_activity" in df.columns:
        df["task_activity_recency_score"] = 1 / (1 + df["days_since_last_task_activity"])
        df["task_activity_recency_score"] = df["task_activity_recency_score"].fillna(0)

    if "days_since_last_task_created" in df.columns:
        df["task_created_recency_score"] = 1 / (1 + df["days_since_last_task_created"])
        df["task_created_recency_score"] = df["task_created_recency_score"].fillna(0)

    return df


def build_model_ready_v6(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert booleans if present
    df = standardize_booleans_to_int(
        df,
        [
            "lead_is_converted",
        ],
    )

    # Clean noisy recency values
    df = safe_clip_nonnegative(
        df,
        [
            "days_since_last_task_created",
            "days_since_last_task_activity",
        ],
    )

    # Recompute recency scores after clipping
    df = recompute_recency_scores(df)

    # Drop noisy / constant / low-value columns
    drop_cols = [
        "owner_tenure_bucket",   # too many missing values / noisy
        "owner_department",      # constant in EDA
    ]

    model_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Put target at the end
    target_col = "lead_is_converted"
    cols = [c for c in model_df.columns if c != target_col] + [target_col]
    model_df = model_df[cols]

    return model_df


def main() -> None:
    df = load_data()
    model_df = build_model_ready_v6(df)

    model_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved model-ready dataset to: {OUTPUT_FILE}")
    print(f"Shape: {model_df.shape}")
    print("Columns:")
    print(model_df.columns.tolist())


if __name__ == "__main__":
    main()