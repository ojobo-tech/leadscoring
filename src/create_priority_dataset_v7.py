from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_model_ready_v6.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_priority_dataset_v7.csv"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)


def to_numeric_clipped(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0)


def inverse_percentile_rank(series: pd.Series) -> pd.Series:
    return 1 - percentile_rank(series)


def add_cleaned_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
        "lead_age_days",
        "owner_tenure_days",
        "days_since_last_task_created",
        "days_since_last_task_activity",
        "open_task_ratio",
        "closed_task_ratio",
        "high_priority_task_ratio",
        "task_velocity",
        "high_priority_velocity",
        "task_count_per_owner_day",
        "task_activity_recency_score",
        "task_created_recency_score",
    ]
    df = to_numeric_clipped(df, numeric_cols)

    # Recompute a cleaner owner-based intensity feature
    if "task_count" in df.columns and "owner_tenure_days" in df.columns:
        df["task_count_per_owner_day_clean"] = df["task_count"] / (df["owner_tenure_days"] + 1)
    else:
        df["task_count_per_owner_day_clean"] = np.nan

    return df


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Engagement: how active and interaction-heavy the lead appears
    engagement_parts = []

    for col in [
        "open_task_ratio",
        "closed_task_ratio",
        "high_priority_task_ratio",
        "task_activity_recency_score",
        "task_created_recency_score",
        "task_velocity",
        "high_priority_velocity",
    ]:
        if col in df.columns:
            engagement_parts.append(percentile_rank(df[col]))

    if engagement_parts:
        df["engagement_score"] = pd.concat(engagement_parts, axis=1).mean(axis=1)
    else:
        df["engagement_score"] = 0.0

    # Urgency: how quickly this lead should be reviewed
    urgency_parts = []

    for col in [
        "days_since_last_task_activity",
        "days_since_last_task_created",
        "lead_age_days",
    ]:
        if col in df.columns:
            urgency_parts.append(inverse_percentile_rank(df[col]))

    if urgency_parts:
        df["urgency_score"] = pd.concat(urgency_parts, axis=1).mean(axis=1)
    else:
        df["urgency_score"] = 0.0

    # Activity strength: total intensity of work around the lead
    activity_parts = []

    for col in [
        "task_count",
        "high_priority_task_count",
        "task_count_per_owner_day_clean",
    ]:
        if col in df.columns:
            activity_parts.append(percentile_rank(df[col]))

    if activity_parts:
        df["activity_strength_score"] = pd.concat(activity_parts, axis=1).mean(axis=1)
    else:
        df["activity_strength_score"] = 0.0

    # Final priority score: combine the three business-friendly signals
    df["priority_score"] = (
        100
        * (
            0.50 * df["engagement_score"]
            + 0.30 * df["urgency_score"]
            + 0.20 * df["activity_strength_score"]
        )
    )

    # Priority bucket
    df["priority_bucket"] = pd.cut(
        df["priority_score"],
        bins=[-0.01, 25, 50, 75, 100],
        labels=["Cold", "Cool", "Warm", "Hot"],
        include_lowest=True,
    ).astype(str)

    # Recommended action
    def action_from_bucket(bucket: str) -> str:
        if bucket == "Hot":
            return "Contact today"
        if bucket == "Warm":
            return "Follow up within 48 hours"
        if bucket == "Cool":
            return "Nurture and monitor"
        return "Add to nurture campaign"

    df["recommended_action"] = df["priority_bucket"].apply(action_from_bucket)

    return df


def build_priority_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean only the fields we need for the scorecard
    df = add_cleaned_features(df)
    df = build_scores(df)

    # Keep the client-facing scorecard focused and readable
    keep_cols = [
        "status",
        "lead_source",
        "lead_owner_id",
        "owner_role",
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
        "lead_age_days",
        "owner_tenure_days",
        "days_since_last_task_created",
        "days_since_last_task_activity",
        "open_task_ratio",
        "closed_task_ratio",
        "high_priority_task_ratio",
        "task_velocity",
        "high_priority_velocity",
        "task_count_per_owner_day_clean",
        "task_activity_recency_score",
        "task_created_recency_score",
        "priority_score",
        "priority_bucket",
        "engagement_score",
        "urgency_score",
        "activity_strength_score",
        "recommended_action",
        "lead_is_converted",  # keep for backtesting only
    ]

    model_df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Put the backtest label at the end
    if "lead_is_converted" in model_df.columns:
        target_col = "lead_is_converted"
        cols = [c for c in model_df.columns if c != target_col] + [target_col]
        model_df = model_df[cols]

    return model_df


def main() -> None:
    df = load_data()
    priority_df = build_priority_dataset(df)

    priority_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved priority dataset to: {OUTPUT_FILE}")
    print(f"Shape: {priority_df.shape}")
    print("Columns:")
    print(priority_df.columns.tolist())


if __name__ == "__main__":
    main()