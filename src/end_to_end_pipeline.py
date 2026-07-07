from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# Make sure the project root is on the import path so we can load src/end_to_end_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.end_to_end_pipeline import run_end_to_end_pipeline


APP_TITLE = "AI Lead Prioritization and Engagement Scoring"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "streamlit_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical fields expected by the automated pipeline.
CANONICAL_FIELDS = [
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

ROLE_KEYWORDS = {
    "opportunity": ["opportunity", "opp"],
    "lead": ["lead", "prospect"],
    "task": ["task"],
    "account": ["account"],
    "user": ["user", "owner", "rep", "salesperson"],
    "contact": ["contact"],
}

# Column alias mapping for common CRM exports.
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
    "companynorm": "company_norm",
    "company_norm": "company_norm",
    "company": "company",
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

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Upload one CSV or several raw CRM CSVs. The system will automatically map, join, score, and explain the leads."
)


@st.cache_resource
def load_pipeline():
    return run_end_to_end_pipeline


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


def aggregate_tasks(tasks_df: pd.DataFrame) -> pd.DataFrame:
    tasks_df = canonicalize_columns(tasks_df)
    tasks_df = tasks_df.copy()

    key_col = _first_existing_col(tasks_df, ["lead_id", "who_id"])
    if key_col is None:
        return pd.DataFrame()

    # Make sure common task fields exist before grouping.
    for col in ["is_closed", "priority", "created_date", "activity_date"]:
        if col not in tasks_df.columns:
            tasks_df[col] = pd.NA

    def _priority_high(series: pd.Series) -> int:
        return int((series.astype(str).str.lower() == "high").sum())

    grouped = (
        tasks_df.dropna(subset=[key_col])
        .groupby(key_col)
        .agg(
            task_count=(key_col, "count"),
            open_task_count=("is_closed", lambda x: (~x.fillna(False)).sum()),
            closed_task_count=("is_closed", lambda x: x.fillna(False).sum()),
            high_priority_task_count=("priority", _priority_high),
            last_task_created_date=("created_date", "max"),
            last_task_activity_date=("activity_date", "max"),
        )
        .reset_index()
        .rename(columns={key_col: "lead_id"})
    )

    return grouped


def build_primary_frame(file_frames: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    """Build one canonical lead-level frame from one or multiple uploaded files."""
    # Choose a primary table.
    primary_order = ["opportunity", "lead", "account", "contact", "unknown"]
    primary_role = next((r for r in primary_order if r in file_frames and len(file_frames[r]) > 0), None)

    if primary_role is None:
        raise ValueError("No readable CSV files were uploaded.")

    base = pd.concat([canonicalize_columns(df) for df in file_frames[primary_role]], ignore_index=True)

    # Merge task aggregates if available.
    if "task" in file_frames:
        task_frames = [canonicalize_columns(df) for df in file_frames["task"]]
        tasks = pd.concat(task_frames, ignore_index=True)
        task_agg = aggregate_tasks(tasks)
        if not task_agg.empty and "lead_id" in base.columns and "lead_id" in task_agg.columns:
            base = base.merge(task_agg, how="left", on="lead_id")

    # Merge user/owner enrichment if available.
    if "user" in file_frames:
        user_frames = [canonicalize_columns(df) for df in file_frames["user"]]
        users = pd.concat(user_frames, ignore_index=True)
        owner_key = _first_existing_col(users, ["lead_owner_id", "owner_id", "user_id", "id"])
        if owner_key is not None:
            users = users.rename(columns={owner_key: "lead_owner_id"})
            keep_cols = [c for c in ["lead_owner_id", "owner_role", "owner_department", "owner_is_active", "owner_hire_date"] if c in users.columns]
            users = users[keep_cols].drop_duplicates(subset=["lead_owner_id"])
            if "lead_owner_id" in base.columns:
                base = base.merge(users, how="left", on="lead_owner_id", suffixes=("", "_user"))

    # Merge account enrichment if available.
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
    """Create the canonical columns expected by the automated scoring pipeline."""
    df = df.copy()

    for col in CANONICAL_FIELDS:
        if col not in df.columns:
            if col in {"status", "lead_source", "lead_owner_id", "owner_role"}:
                df[col] = "Unknown"
            else:
                df[col] = pd.NA

    for col in ["status", "lead_source", "lead_owner_id", "owner_role"]:
        df[col] = df[col].apply(clean_text)

    # Numeric cleaning.
    numeric_cols = [c for c in CANONICAL_FIELDS if c not in {"status", "lead_source", "lead_owner_id", "owner_role"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive missing features from available raw data.
    if df["task_count_per_owner_day_clean"].isna().all() and {"task_count", "owner_tenure_days"}.issubset(df.columns):
        df["task_count_per_owner_day_clean"] = df["task_count"] / (df["owner_tenure_days"].fillna(0) + 1)

    if df["task_velocity"].isna().all() and {"task_count", "lead_age_days"}.issubset(df.columns):
        df["task_velocity"] = df["task_count"] / (df["lead_age_days"].fillna(0) + 1)

    if df["high_priority_velocity"].isna().all() and {"high_priority_task_count", "lead_age_days"}.issubset(df.columns):
        df["high_priority_velocity"] = df["high_priority_task_count"] / (df["lead_age_days"].fillna(0) + 1)

    if df["open_task_ratio"].isna().all() and {"open_task_count", "task_count"}.issubset(df.columns):
        denom = df["task_count"].replace(0, pd.NA)
        df["open_task_ratio"] = (df["open_task_count"] / denom).fillna(0)

    if df["closed_task_ratio"].isna().all() and {"closed_task_count", "task_count"}.issubset(df.columns):
        denom = df["task_count"].replace(0, pd.NA)
        df["closed_task_ratio"] = (df["closed_task_count"] / denom).fillna(0)

    if df["high_priority_task_ratio"].isna().all() and {"high_priority_task_count", "task_count"}.issubset(df.columns):
        denom = df["task_count"].replace(0, pd.NA)
        df["high_priority_task_ratio"] = (df["high_priority_task_count"] / denom).fillna(0)

    # Recency fields if date columns exist.
    if "last_task_activity_date" in df.columns and df["days_since_last_task_activity"].isna().all():
        df["last_task_activity_date"] = pd.to_datetime(df["last_task_activity_date"], errors="coerce")
        ref = df["last_task_activity_date"].max()
        if pd.notna(ref):
            df["days_since_last_task_activity"] = (ref - df["last_task_activity_date"]).dt.days.fillna(0).clip(lower=0)

    if "last_task_created_date" in df.columns and df["days_since_last_task_created"].isna().all():
        df["last_task_created_date"] = pd.to_datetime(df["last_task_created_date"], errors="coerce")
        ref = df["last_task_created_date"].max()
        if pd.notna(ref):
            df["days_since_last_task_created"] = (ref - df["last_task_created_date"]).dt.days.fillna(0).clip(lower=0)

    # If the owner role is still missing, leave it as Unknown.
    df["owner_role"] = df["owner_role"].fillna("Unknown").astype(str)

    return df


st.sidebar.header("Upload raw data")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
)
run_button = st.sidebar.button("Score Leads", type="primary", use_container_width=True)

if uploaded_files:
    st.sidebar.markdown("### Detected file roles")
    for f in uploaded_files:
        st.sidebar.write(f"- {f.name} → {detect_role(f.name)}")

st.markdown(
    """
    The system will automatically:
    - detect file roles from names
    - map common CRM column aliases
    - join compatible raw files
    - standardize and engineer features
    - score leads
    - generate explanations
    - return a downloadable scored file
    """
)

if not uploaded_files:
    st.info("Upload one or more raw CSVs to begin.")
    st.stop()

# Read all uploaded files.
file_frames: Dict[str, List[pd.DataFrame]] = {}
for file in uploaded_files:
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        st.error(f"Could not read {file.name}: {exc}")
        st.stop()
    role = detect_role(file.name)
    file_frames.setdefault(role, []).append(df)

# Show a light preview of the first uploaded file.
first_preview = next(iter(file_frames.values()))[0]
st.subheader("Uploaded file preview")
st.dataframe(first_preview.head(10), use_container_width=True)

if not run_button:
    st.caption("Click **Score Leads** to run the automated pipeline.")
    st.stop()

# Build canonical frame from one or multiple uploads.
try:
    merged_raw = build_primary_frame(file_frames)
    merged_raw = ensure_canonical_features(merged_raw)
except Exception as exc:
    st.error(f"Could not build the automated input frame: {exc}")
    st.stop()

# Save the merged data to a temporary file and run the automated pipeline.
with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp_path = Path(tmp.name)
    merged_raw.to_csv(tmp_path, index=False)

try:
    scored_df = run_end_to_end_pipeline(
        input_csv_path=tmp_path,
        output_csv_path=OUTPUT_DIR / "scored_leads.csv",
    )
except Exception as exc:
    st.error(f"Pipeline failed: {exc}")
    st.stop()
finally:
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

# Remove duplicate columns defensively before displaying.
scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()

st.success(f"Scored {len(scored_df):,} leads successfully.")

hot_count = int((scored_df.get("priority_bucket", pd.Series(dtype=str)) == "Hot").sum())
warm_count = int((scored_df.get("priority_bucket", pd.Series(dtype=str)) == "Warm").sum())
cool_count = int((scored_df.get("priority_bucket", pd.Series(dtype=str)) == "Cool").sum())
cold_count = int((scored_df.get("priority_bucket", pd.Series(dtype=str)) == "Cold").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Leads scored", f"{len(scored_df):,}")
m2.metric("Hot leads", f"{hot_count:,}")
m3.metric("Warm leads", f"{warm_count:,}")
m4.metric("Cold leads", f"{cold_count:,}")

st.subheader("Priority distribution")
st.bar_chart(pd.Series({"Hot": hot_count, "Warm": warm_count, "Cool": cool_count, "Cold": cold_count}))

st.subheader("Ranked leads")
rank_cols = [
    c
    for c in [
        "rank",
        "lead_owner_id",
        "lead_source",
        "status",
        "owner_role",
        "priority_score",
        "priority_bucket",
        "recommended_action",
        "reason_1",
        "reason_2",
        "reason_3",
        "xai_summary",
        "score_source",
    ]
    if c in scored_df.columns
]

st.dataframe(
    scored_df.sort_values("priority_score", ascending=False)[rank_cols],
    use_container_width=True,
    height=450,
)

st.download_button(
    label="Download scored CSV",
    data=scored_df.to_csv(index=False).encode("utf-8"),
    file_name="scored_leads.csv",
    mime="text/csv",
    use_container_width=True,
)

st.subheader("Lead explorer")
ranked = scored_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
lead_options = [
    f"{i+1}. {ranked.loc[i, 'lead_owner_id'] if 'lead_owner_id' in ranked.columns else 'Lead'} | {ranked.loc[i, 'priority_bucket']} | {float(ranked.loc[i, 'priority_score']):.1f}"
    for i in range(len(ranked))
]
selected_label = st.selectbox("Select a lead to inspect", lead_options, index=0)
selected_idx = lead_options.index(selected_label)
selected = ranked.loc[selected_idx]

left, right = st.columns([1.1, 0.9])
with left:
    st.markdown("### Selected lead details")
    st.write(selected.to_dict())

with right:
    st.markdown("### Explanation")
    st.write(f"**Bucket:** {selected.get('priority_bucket', 'Cold')}")
    st.write(f"**Recommended action:** {selected.get('recommended_action', 'Review')}")
    st.write(f"**Reason 1:** {selected.get('reason_1', '')}")
    st.write(f"**Reason 2:** {selected.get('reason_2', '')}")
    st.write(f"**Reason 3:** {selected.get('reason_3', '')}")
    if "xai_summary" in selected:
        st.info(selected.get("xai_summary", ""))

st.caption(
    "This dashboard now supports one raw CSV or multiple raw CSVs. The client only uploads files and receives scored, explained output."
)
