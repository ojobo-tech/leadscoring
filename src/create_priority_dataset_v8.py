from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "lead_model_ready_v6.csv"
OUTPUT_FILE = PROCESSED_DIR / "lead_priority_dataset_v8.csv"
TARGET_COL = "lead_is_converted"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)


def percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(pct=True, method="average").fillna(0)


def inverse_percentile_rank(series: pd.Series) -> pd.Series:
    return 1 - percentile_rank(series)


def to_numeric_clipped(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def smoothed_target_encoding(
    df: pd.DataFrame,
    col: str,
    target_col: str = TARGET_COL,
    m: float = 50.0,
) -> pd.Series:
    """
    Smoothed historical conversion rate by category.
    This gives business context without making the score too noisy.
    """
    if col not in df.columns or target_col not in df.columns:
        return pd.Series([df[target_col].mean()] * len(df), index=df.index)

    overall_rate = df[target_col].mean()

    stats = (
        df.groupby(col)[target_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "group_mean", "count": "group_count"})
    )

    stats["smoothed_rate"] = (
        (stats["group_count"] * stats["group_mean"]) + (m * overall_rate)
    ) / (stats["group_count"] + m)

    return df[col].map(stats["smoothed_rate"]).fillna(overall_rate)


def build_context_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Smoothed business context scores
    df["status_context_score_raw"] = smoothed_target_encoding(df, "status")
    df["lead_source_context_score_raw"] = smoothed_target_encoding(df, "lead_source")
    df["owner_role_context_score_raw"] = smoothed_target_encoding(df, "owner_role")

    # Rank them so the score has better spread
    df["status_context_score"] = percentile_rank(df["status_context_score_raw"])
    df["lead_source_context_score"] = percentile_rank(df["lead_source_context_score_raw"])
    df["owner_role_context_score"] = percentile_rank(df["owner_role_context_score_raw"])

    # Weighted context score
    df["context_score"] = (
        0.45 * df["status_context_score"]
        + 0.35 * df["lead_source_context_score"]
        + 0.20 * df["owner_role_context_score"]
    )

    return df


def build_engagement_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure this helper exists even if not present in the input file
    if "task_count_per_owner_day_clean" not in df.columns:
        if "task_count" in df.columns and "owner_tenure_days" in df.columns:
            df["task_count_per_owner_day_clean"] = df["task_count"] / (df["owner_tenure_days"] + 1)
        else:
            df["task_count_per_owner_day_clean"] = 0

    df["engagement_score"] = (
        0.25 * percentile_rank(df["task_count"])
        + 0.20 * percentile_rank(df["high_priority_task_count"])
        + 0.15 * percentile_rank(df["open_task_count"])
        + 0.15 * percentile_rank(df["closed_task_count"])
        + 0.15 * percentile_rank(df["task_activity_recency_score"])
        + 0.10 * percentile_rank(df["task_created_recency_score"])
    )

    df["urgency_score"] = (
        0.45 * inverse_percentile_rank(df["days_since_last_task_activity"])
        + 0.35 * inverse_percentile_rank(df["days_since_last_task_created"])
        + 0.20 * inverse_percentile_rank(df["lead_age_days"])
    )

    df["activity_strength_score"] = (
        0.40 * percentile_rank(df["task_velocity"])
        + 0.25 * percentile_rank(df["high_priority_velocity"])
        + 0.20 * percentile_rank(df["task_count_per_owner_day_clean"])
        + 0.15 * percentile_rank(df["high_priority_task_ratio"])
    )

    return df


def build_priority_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Final priority score
    df["priority_score"] = 100 * (
        0.40 * df["context_score"]
        + 0.30 * df["urgency_score"]
        + 0.20 * df["engagement_score"]
        + 0.10 * df["activity_strength_score"]
    )

    # Priority percentile and bucket
    df["priority_percentile"] = percentile_rank(df["priority_score"])

    df["priority_bucket"] = pd.cut(
        df["priority_percentile"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Cold", "Cool", "Warm", "Hot"],
        include_lowest=True,
    ).astype(str)

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

    # Make sure numeric fields are numeric
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
        "task_count_per_owner_day_clean",
        "task_activity_recency_score",
        "task_created_recency_score",
    ]
    df = to_numeric_clipped(df, numeric_cols)

    # Create missing helper if needed
    if "task_count_per_owner_day_clean" not in df.columns:
        if "task_count" in df.columns and "owner_tenure_days" in df.columns:
            df["task_count_per_owner_day_clean"] = df["task_count"] / (df["owner_tenure_days"] + 1)
        else:
            df["task_count_per_owner_day_clean"] = 0

    # Build context and scoring layers
    df = build_context_scores(df)
    df = build_engagement_scores(df)
    df = build_priority_score(df)

    # Keep the scorecard readable
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
        "status_context_score",
        "lead_source_context_score",
        "owner_role_context_score",
        "context_score",
        "engagement_score",
        "urgency_score",
        "activity_strength_score",
        "priority_score",
        "priority_percentile",
        "priority_bucket",
        "recommended_action",
        TARGET_COL,
    ]

    scorecard_df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Put target at the end for backtesting only
    if TARGET_COL in scorecard_df.columns:
        cols = [c for c in scorecard_df.columns if c != TARGET_COL] + [TARGET_COL]
        scorecard_df = scorecard_df[cols]

    return scorecard_df


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