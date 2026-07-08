from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import joblib  # optional; pipeline still works without a saved model artifact
except Exception:  # pragma: no cover
    joblib = None

MODEL_FILE = Path("models/lead_scoring_model.joblib")
OUTPUT_DIR = Path("reports/automated_pipeline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_KEYWORDS = {
    "opportunity": ["opportunity", "opp"],
    "lead": ["lead", "prospect"],
    "task": ["task"],
    "account": ["account"],
    "user": ["user", "owner", "rep", "salesperson"],
    "contact": ["contact"],
}

ALIAS_MAP = {
    "status": "status",
    "leadstatus": "status",
    "lead_status": "status",
    "stage": "status",
    "leadstage": "status",
    "lead_stage": "status",
    "leadsource": "lead_source",
    "lead_source": "lead_source",
    "source": "lead_source",
    "channel": "lead_source",
    "leadownerid": "lead_owner_id",
    "ownerid": "lead_owner_id",
    "owner_id": "lead_owner_id",
    "assignedto": "lead_owner_id",
    "owner": "lead_owner_id",
    "rep": "lead_owner_id",
    "ownerrole": "owner_role",
    "owner_role": "owner_role",
    "role": "owner_role",
    "rep_role": "owner_role",
    "sales_role": "owner_role",
    "taskcount": "task_count",
    "task_count": "task_count",
    "tasks": "task_count",
    "opentaskcount": "open_task_count",
    "open_task_count": "open_task_count",
    "open_tasks": "open_task_count",
    "closedtaskcount": "closed_task_count",
    "closed_task_count": "closed_task_count",
    "closed_tasks": "closed_task_count",
    "highprioritytaskcount": "high_priority_task_count",
    "high_priority_task_count": "high_priority_task_count",
    "priority_tasks": "high_priority_task_count",
    "leadagedays": "lead_age_days",
    "lead_age_days": "lead_age_days",
    "agedays": "lead_age_days",
    "ownertenuredays": "owner_tenure_days",
    "owner_tenure_days": "owner_tenure_days",
    "tenuredays": "owner_tenure_days",
    "dayssincelasttaskcreated": "days_since_last_task_created",
    "days_since_last_task_created": "days_since_last_task_created",
    "lasttaskcreateddays": "days_since_last_task_created",
    "dayssincelasttaskactivity": "days_since_last_task_activity",
    "days_since_last_task_activity": "days_since_last_task_activity",
    "lasttaskactivitydays": "days_since_last_task_activity",
    "opentaskratio": "open_task_ratio",
    "open_task_ratio": "open_task_ratio",
    "closedtaskratio": "closed_task_ratio",
    "closed_task_ratio": "closed_task_ratio",
    "highprioritytaskratio": "high_priority_task_ratio",
    "high_priority_task_ratio": "high_priority_task_ratio",
    "taskvelocity": "task_velocity",
    "task_velocity": "task_velocity",
    "highpriorityvelocity": "high_priority_velocity",
    "high_priority_velocity": "high_priority_velocity",
    "taskcountperownerdayclean": "task_count_per_owner_day_clean",
    "task_count_per_owner_day_clean": "task_count_per_owner_day_clean",
    "taskcountperownerday": "task_count_per_owner_day_clean",
    "taskactivityrecencyscore": "task_activity_recency_score",
    "task_activity_recency_score": "task_activity_recency_score",
    "taskcreatedrecencyscore": "task_created_recency_score",
    "task_created_recency_score": "task_created_recency_score",
    "accountid": "account_id",
    "account_id": "account_id",
    "convertedaccountid": "converted_account_id",
    "converted_account_id": "converted_account_id",
    "userid": "user_id",
    "user_id": "user_id",
    "whoid": "who_id",
    "who_id": "who_id",
    "leadid": "lead_id",
    "lead_id": "lead_id",
}

STATUS_WEIGHTS = {
    "qualified": 0.90,
    "working": 0.75,
    "open": 0.65,
    "nurturing": 0.40,
}
SOURCE_WEIGHTS = {
    "partner referral": 0.90,
    "event": 0.85,
    "web": 0.75,
    "inbound": 0.78,
    "trade show": 0.70,
    "outbound": 0.50,
}
ROLE_WEIGHTS = {
    "ae": 0.88,
    "bdr": 0.72,
    "sdr": 0.68,
}

BUCKET_ACTIONS = {
    "Hot": "Contact today",
    "Warm": "Follow up within 48 hours",
    "Cool": "Nurture and monitor",
    "Cold": "Add to nurture campaign",
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def detect_role(filename: str) -> str:
    lower = filename.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return role
    return "unknown"


def clean_text(value) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        norm = normalize_name(col)
        if norm in ALIAS_MAP:
            rename_map[col] = ALIAS_MAP[norm]
    df = df.rename(columns=rename_map)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_upload_summary(uploaded_files: List) -> pd.DataFrame:
    rows = []
    for f in uploaded_files:
        try:
            preview = pd.read_csv(f)
            rows.append(
                {
                    "file": f.name,
                    "role": detect_role(f.name),
                    "rows": len(preview),
                    "columns": len(preview.columns),
                    "sample_columns": ", ".join(list(preview.columns[:6])),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": f.name,
                    "role": detect_role(f.name),
                    "rows": "-",
                    "columns": "-",
                    "sample_columns": f"Unreadable: {exc}",
                }
            )
    return pd.DataFrame(rows)


def lead_label(row: pd.Series) -> str:
    if "name" in row and pd.notna(row["name"]):
        base = str(row["name"])
    elif "company" in row and pd.notna(row["company"]):
        base = str(row["company"])
    elif "lead_id" in row and pd.notna(row["lead_id"]):
        base = str(row["lead_id"])
    elif "lead_owner_id" in row and pd.notna(row["lead_owner_id"]):
        base = str(row["lead_owner_id"])
    else:
        base = "Lead"

    bucket = row.get("priority_bucket", "Cold")
    score = row.get("priority_score", row.get("model_probability", 0))
    return f"{base} | {bucket} | {float(score):.1f}"


def aggregate_tasks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    tasks_df = canonicalize_columns(tasks_df).copy()

    key_col = _first_existing_col(tasks_df, ["lead_id", "who_id"])
    if key_col is None:
        return pd.DataFrame()

    for col in ["status", "is_closed", "priority", "created_date", "activity_date"]:
        if col not in tasks_df.columns:
            tasks_df[col] = pd.NA

    def _closed_mask(df: pd.DataFrame) -> pd.Series:
        if df["is_closed"].notna().any():
            s = df["is_closed"].astype(str).str.lower()
            return s.isin(["true", "1", "yes", "y", "closed", "done", "complete"])
        s = df["status"].astype(str).str.lower()
        return s.str.contains("closed|won|done|complete", na=False)

    def _priority_high(series: pd.Series) -> int:
        return int((series.astype(str).str.lower() == "high").sum())

    closed_mask = _closed_mask(tasks_df)

    grouped = (
        tasks_df.dropna(subset=[key_col])
        .assign(_closed=closed_mask)
        .groupby(key_col)
        .agg(
            task_count=(key_col, "count"),
            open_task_count=("_closed", lambda x: (~x).sum()),
            closed_task_count=("_closed", "sum"),
            high_priority_task_count=("priority", _priority_high),
            last_task_created_date=("created_date", "max"),
            last_task_activity_date=("activity_date", "max"),
        )
        .reset_index()
        .rename(columns={key_col: "lead_id"})
    )

    return grouped


def build_primary_frame(file_frames: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    primary_order = ["opportunity", "lead", "account", "contact", "unknown"]
    primary_role = next((r for r in primary_order if r in file_frames and len(file_frames[r]) > 0), None)

    if primary_role is None:
        raise ValueError("No readable CSV files were uploaded.")

    base = pd.concat([canonicalize_columns(df) for df in file_frames[primary_role]], ignore_index=True)

    if "task" in file_frames:
        task_frames = [canonicalize_columns(df) for df in file_frames["task"]]
        tasks = pd.concat(task_frames, ignore_index=True)
        task_agg = aggregate_tasks(tasks)
        if not task_agg.empty:
            if "lead_id" in base.columns:
                base = base.merge(task_agg, how="left", on="lead_id")
            elif "who_id" in base.columns:
                base = base.merge(
                    task_agg,
                    how="left",
                    left_on="who_id",
                    right_on="lead_id",
                    suffixes=("", "_task"),
                )

    if "user" in file_frames:
        user_frames = [canonicalize_columns(df) for df in file_frames["user"]]
        users = pd.concat(user_frames, ignore_index=True)
        owner_key = _first_existing_col(users, ["lead_owner_id", "owner_id", "user_id", "id"])
        if owner_key is not None:
            users = users.rename(columns={owner_key: "lead_owner_id"})
            keep_cols = [
                c
                for c in [
                    "lead_owner_id",
                    "owner_role",
                    "owner_department",
                    "owner_is_active",
                    "owner_hire_date",
                ]
                if c in users.columns
            ]
            users = users[keep_cols].drop_duplicates(subset=["lead_owner_id"])
            if "lead_owner_id" in base.columns:
                base = base.merge(users, how="left", on="lead_owner_id", suffixes=("", "_user"))

    if "account" in file_frames:
        account_frames = [canonicalize_columns(df) for df in file_frames["account"]]
        accounts = pd.concat(account_frames, ignore_index=True)
        account_left = _first_existing_col(base, ["converted_account_id", "account_id"])
        account_right = _first_existing_col(accounts, ["account_id", "converted_account_id"])
        if account_left and account_right:
            accounts = accounts.rename(columns={account_right: account_left})
            keep_cols = [c for c in accounts.columns if c != account_left]
            accounts = accounts[[account_left] + keep_cols].drop_duplicates(subset=[account_left])
            base = base.merge(accounts, how="left", on=account_left, suffixes=("", "_account"))

    return base


def ensure_canonical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    canonical_fields = [
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
    ]

    for col in canonical_fields:
        if col not in df.columns:
            df[col] = "Unknown" if col in {"status", "lead_source", "lead_owner_id", "owner_role"} else pd.NA

    for col in ["status", "lead_source", "lead_owner_id", "owner_role"]:
        df[col] = df[col].apply(clean_text)

    numeric_cols = [c for c in canonical_fields if c not in {"status", "lead_source", "lead_owner_id", "owner_role"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["task_count_per_owner_day_clean"].isna().all():
        df["task_count_per_owner_day_clean"] = df["task_count"] / (df["owner_tenure_days"].fillna(0) + 1)

    if df["task_velocity"].isna().all():
        df["task_velocity"] = df["task_count"] / (df["lead_age_days"].fillna(0) + 1)

    if df["high_priority_velocity"].isna().all():
        df["high_priority_velocity"] = df["high_priority_task_count"] / (df["lead_age_days"].fillna(0) + 1)

    if df["open_task_ratio"].isna().all():
        denom = df["task_count"].replace(0, pd.NA)
        df["open_task_ratio"] = (df["open_task_count"] / denom).fillna(0)

    if df["closed_task_ratio"].isna().all():
        denom = df["task_count"].replace(0, pd.NA)
        df["closed_task_ratio"] = (df["closed_task_count"] / denom).fillna(0)

    if df["high_priority_task_ratio"].isna().all():
        denom = df["task_count"].replace(0, pd.NA)
        df["high_priority_task_ratio"] = (df["high_priority_task_count"] / denom).fillna(0)

    if "last_task_activity_date" in df.columns:
        df["last_task_activity_date"] = pd.to_datetime(df["last_task_activity_date"], errors="coerce")
        ref = df["last_task_activity_date"].max()
        if pd.notna(ref) and df["days_since_last_task_activity"].isna().all():
            df["days_since_last_task_activity"] = (
                ref - df["last_task_activity_date"]
            ).dt.days.fillna(0).clip(lower=0)

    if "last_task_created_date" in df.columns:
        df["last_task_created_date"] = pd.to_datetime(df["last_task_created_date"], errors="coerce")
        ref = df["last_task_created_date"].max()
        if pd.notna(ref) and df["days_since_last_task_created"].isna().all():
            df["days_since_last_task_created"] = (
                ref - df["last_task_created_date"]
            ).dt.days.fillna(0).clip(lower=0)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df[canonical_fields + [c for c in df.columns if c not in canonical_fields]].copy()


def _batch_percentile(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    fill = s.median() if s.notna().any() else 0
    return s.fillna(fill).rank(pct=True, method="average").fillna(0.5)


def _inverse_percentile(series: pd.Series) -> pd.Series:
    return 1.0 - _batch_percentile(series)


def _maybe_predict_probability(df: pd.DataFrame) -> Optional[pd.Series]:
    if joblib is None or not MODEL_FILE.exists():
        return None

    try:
        artifact = joblib.load(MODEL_FILE)
    except Exception:
        return None

    try:
        if hasattr(artifact, "predict_proba"):
            proba = artifact.predict_proba(df)[:, 1]
            return pd.Series(proba, index=df.index, name="model_probability")

        if hasattr(artifact, "named_steps") and "preprocessor" in artifact.named_steps and "model" in artifact.named_steps:
            X = artifact.named_steps["preprocessor"].transform(df)
            proba = artifact.named_steps["model"].predict_proba(X)[:, 1]
            return pd.Series(proba, index=df.index, name="model_probability")
    except Exception:
        return None

    return None


def _make_reasons(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    candidates = pd.DataFrame(index=df.index)

    candidates["lower days since last task activity increased priority"] = _inverse_percentile(
        df["days_since_last_task_activity"]
    )
    candidates["lower days since last task created increased priority"] = _inverse_percentile(
        df["days_since_last_task_created"]
    )
    candidates["higher task activity recency score increased priority"] = _batch_percentile(
        df["task_activity_recency_score"]
    )
    candidates["higher task created recency score increased priority"] = _batch_percentile(
        df["task_created_recency_score"]
    )
    candidates["higher closed task count increased priority"] = _batch_percentile(
        df["closed_task_count"]
    )
    candidates["higher high priority task count increased priority"] = _batch_percentile(
        df["high_priority_task_count"]
    )
    candidates["higher task velocity increased priority"] = _batch_percentile(
        df["task_velocity"]
    )
    candidates["higher high priority velocity increased priority"] = _batch_percentile(
        df["high_priority_velocity"]
    )
    candidates["higher task count per owner day increased priority"] = _batch_percentile(
        df["task_count_per_owner_day_clean"]
    )
    candidates["higher lead source context score increased priority"] = _batch_percentile(
        df["lead_source_context_score"]
    )
    candidates["higher owner role context score increased priority"] = _batch_percentile(
        df["owner_role_context_score"]
    )

    reason_1, reason_2, reason_3, summary = [], [], [], []
    for idx in candidates.index:
        s = candidates.loc[idx].sort_values(ascending=False)
        top = [str(x) for x in s.index[:3]]
        reason_1.append(top[0] if len(top) > 0 else "No strong driver detected")
        reason_2.append(top[1] if len(top) > 1 else "")
        reason_3.append(top[2] if len(top) > 2 else "")

        if len(top) >= 3:
            summary.append(f"This lead is ranked higher because {top[0]}, {top[1]}, and {top[2]}.")
        elif len(top) == 2:
            summary.append(f"This lead is ranked higher because {top[0]} and {top[1]}.")
        else:
            summary.append(f"This lead is ranked higher because {top[0]}.")

    return (
        pd.Series(reason_1, index=df.index, name="reason_1"),
        pd.Series(reason_2, index=df.index, name="reason_2"),
        pd.Series(reason_3, index=df.index, name="reason_3"),
        pd.Series(summary, index=df.index, name="xai_summary"),
    )


def score_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["status_context_score"] = df["status"].astype(str).str.lower().map(STATUS_WEIGHTS).fillna(0.55)
    df["lead_source_context_score"] = df["lead_source"].astype(str).str.lower().map(SOURCE_WEIGHTS).fillna(0.55)
    df["owner_role_context_score"] = df["owner_role"].astype(str).str.lower().map(ROLE_WEIGHTS).fillna(0.55)

    df["context_score"] = (
        0.40 * df["status_context_score"]
        + 0.35 * df["lead_source_context_score"]
        + 0.25 * df["owner_role_context_score"]
    )

    task_count_pct = _batch_percentile(df["task_count"])
    open_task_pct = _batch_percentile(df["open_task_count"])
    closed_task_pct = _batch_percentile(df["closed_task_count"])
    high_priority_pct = _batch_percentile(df["high_priority_task_count"])
    task_velocity_pct = _batch_percentile(df["task_velocity"])
    high_priority_velocity_pct = _batch_percentile(df["high_priority_velocity"])
    count_per_owner_day_pct = _batch_percentile(df["task_count_per_owner_day_clean"])
    activity_recency_pct = _batch_percentile(df["task_activity_recency_score"])
    created_recency_pct = _batch_percentile(df["task_created_recency_score"])

    df["engagement_score"] = (
        0.22 * task_count_pct
        + 0.16 * open_task_pct
        + 0.16 * closed_task_pct
        + 0.16 * high_priority_pct
        + 0.15 * activity_recency_pct
        + 0.15 * created_recency_pct
    ).clip(0, 1)

    urgency_activity_inv = _inverse_percentile(df["days_since_last_task_activity"])
    urgency_created_inv = _inverse_percentile(df["days_since_last_task_created"])
    urgency_age_inv = _inverse_percentile(df["lead_age_days"])

    df["urgency_score"] = (
        0.45 * urgency_activity_inv
        + 0.35 * urgency_created_inv
        + 0.20 * urgency_age_inv
    ).clip(0, 1)

    df["activity_strength_score"] = (
        0.34 * task_velocity_pct
        + 0.28 * high_priority_velocity_pct
        + 0.20 * count_per_owner_day_pct
        + 0.18 * high_priority_pct
    ).clip(0, 1)

    df["priority_score"] = (
        100
        * (
            0.35 * df["context_score"]
            + 0.30 * df["urgency_score"]
            + 0.20 * df["engagement_score"]
            + 0.15 * df["activity_strength_score"]
        )
    ).clip(0, 100)

    df["priority_percentile"] = df["priority_score"].rank(pct=True, method="average").fillna(0.5)

    df["priority_bucket"] = pd.cut(
        df["priority_percentile"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Cold", "Cool", "Warm", "Hot"],
        include_lowest=True,
    ).astype(str)

    df["recommended_action"] = df["priority_bucket"].map(BUCKET_ACTIONS).fillna("Review")

    model_prob = _maybe_predict_probability(df)
    df["model_probability"] = model_prob if model_prob is not None else (df["priority_score"] / 100.0)

    reason_1, reason_2, reason_3, xai_summary = _make_reasons(df)
    df["reason_1"] = reason_1
    df["reason_2"] = reason_2
    df["reason_3"] = reason_3
    df["xai_summary"] = xai_summary

    return df


def _build_final_frame(raw_df: pd.DataFrame, scored_df: pd.DataFrame) -> pd.DataFrame:
    final = scored_df.reset_index(drop=True).copy()
    raw_df = raw_df.reset_index(drop=True)

    for col in raw_df.columns:
        if col not in final.columns:
            final[col] = raw_df[col].values

    ordered_cols = list(raw_df.columns) + [c for c in final.columns if c not in raw_df.columns]
    final = final[ordered_cols]
    final = final.loc[:, ~final.columns.duplicated()].copy()
    return final


def run_end_to_end_pipeline(
    input_csv_path: str | Path,
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    input_csv_path = Path(input_csv_path)
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_csv_path}")

    raw_df = pd.read_csv(input_csv_path)
    raw_df = canonicalize_columns(raw_df)
    prepared_df = ensure_canonical_features(raw_df)
    scored_df = score_frame(prepared_df)
    scored_df = scored_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    scored_df["rank"] = np.arange(1, len(scored_df) + 1)

    final_df = _build_final_frame(raw_df, scored_df)

    if output_csv_path is not None:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv_path, index=False)

    return final_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the automated lead scoring pipeline.")
    parser.add_argument("input_csv", type=str, help="Path to the raw CSV")
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(OUTPUT_DIR / "scored_leads.csv"),
        help="Where to save the scored CSV",
    )
    args = parser.parse_args()

    out = run_end_to_end_pipeline(args.input_csv, args.output_csv)
    print(f"Saved scored output to: {args.output_csv}")
    print(out.head(10).to_string(index=False))