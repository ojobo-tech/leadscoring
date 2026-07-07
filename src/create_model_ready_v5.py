from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_model_ready_v4_with_owner.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_model_ready_v5.csv"

TARGET_COL = "lead_is_converted"

def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)

def safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan) if hasattr(denominator, "replace") else denominator
    return numerator / denominator

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert numeric fields safely
    numeric_cols = [
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
        "lead_age_days",
        "owner_tenure_days",
        "days_since_last_task_created",
        "days_since_last_task_activity",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ratios
    if {"task_count", "open_task_count"}.issubset(df.columns):
        df["open_task_ratio"] = safe_divide(df["open_task_count"], df["task_count"]).fillna(0)

    if {"task_count", "closed_task_count"}.issubset(df.columns):
        df["closed_task_ratio"] = safe_divide(df["closed_task_count"], df["task_count"]).fillna(0)

    if {"task_count", "high_priority_task_count"}.issubset(df.columns):
        df["high_priority_task_ratio"] = safe_divide(df["high_priority_task_count"], df["task_count"]).fillna(0)

    # Pace / intensity
    if {"task_count", "lead_age_days"}.issubset(df.columns):
        df["task_velocity"] = safe_divide(df["task_count"], df["lead_age_days"] + 1).fillna(0)

    if {"high_priority_task_count", "lead_age_days"}.issubset(df.columns):
        df["high_priority_velocity"] = safe_divide(df["high_priority_task_count"], df["lead_age_days"] + 1).fillna(0)

    if {"task_count", "owner_tenure_days"}.issubset(df.columns):
        df["task_count_per_owner_day"] = safe_divide(df["task_count"], df["owner_tenure_days"] + 1).fillna(0)

    # Recency scores
    if "days_since_last_task_activity" in df.columns:
        df["task_activity_recency_score"] = 1 / (1 + df["days_since_last_task_activity"].clip(lower=0))
        df["task_activity_recency_score"] = df["task_activity_recency_score"].fillna(0)

    if "days_since_last_task_created" in df.columns:
        df["task_created_recency_score"] = 1 / (1 + df["days_since_last_task_created"].clip(lower=0))
        df["task_created_recency_score"] = df["task_created_recency_score"].fillna(0)

    # Buckets
    if "lead_age_days" in df.columns:
        df["lead_age_bucket"] = pd.cut(
            df["lead_age_days"],
            bins=[-1, 30, 90, 180, 365, 730, 2000],
            labels=["0_30", "31_90", "91_180", "181_365", "366_730", "730_plus"],
        ).astype(str)

    if "owner_tenure_days" in df.columns:
        df["owner_tenure_bucket"] = pd.cut(
            df["owner_tenure_days"],
            bins=[-1, 30, 90, 180, 365, 730, 3000],
            labels=["0_30", "31_90", "91_180", "181_365", "366_730", "730_plus"],
        ).astype(str)

    if "task_count" in df.columns:
        df["task_count_bucket"] = pd.cut(
            df["task_count"],
            bins=[-1, 0, 2, 5, 10, 100],
            labels=["0", "1_2", "3_5", "6_10", "10_plus"],
        ).astype(str)

    return df

def main() -> None:
    df = load_data()
    df = engineer_features(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved feature-engineered dataset to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()