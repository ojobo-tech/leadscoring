from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.end_to_end_pipeline import run_end_to_end_pipeline


APP_TITLE = "AI Lead Prioritization and Engagement Scoring"
APP_SUBTITLE = (
    "Upload one CSV or multiple CSVs. The system will automatically map, join, "
    "score, and explain the leads."
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "streamlit_outputs"
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


st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📊")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def detect_role(filename: str) -> str:
    lower = filename.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return role
    return "unknown"


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        norm = normalize_name(col)
        if norm in ALIAS_MAP:
            rename_map[col] = ALIAS_MAP[norm]
    return df.rename(columns=rename_map)


def display_value(value):
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return int(value) if value.is_integer() else round(value, 4)
    return value


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
        finally:
            try:
                f.seek(0)
            except Exception:
                pass
    return pd.DataFrame(rows)


def build_primary_frame(file_frames: Dict[str, List[pd.DataFrame]]) -> pd.DataFrame:
    primary_order = ["opportunity", "lead", "account", "contact", "unknown"]
    primary_role = next((r for r in primary_order if r in file_frames and len(file_frames[r]) > 0), None)
    if primary_role is None:
        raise ValueError("No usable CSV files were uploaded.")

    base = pd.concat([canonicalize_columns(df) for df in file_frames[primary_role]], ignore_index=True)

    # Keep only a sensible set of columns if the upload contains lots of raw CRM fields.
    # The scoring pipeline will use what it recognizes and ignore the rest.
    return base


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


st.markdown(
    f"""
    <div style="padding:1.1rem 1.25rem; border-radius:1.2rem; background:linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color:white; margin-bottom:1rem;">
        <div style="font-size:2rem; font-weight:700; line-height:1.1;">{APP_TITLE}</div>
        <div style="margin-top:0.4rem; font-size:0.98rem; opacity:0.92;">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload raw data")
    uploaded_files = st.file_uploader(
        "Choose one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )
    run_button = st.button("Score Leads", type="primary", use_container_width=True)
    st.markdown(
        """
        **What happens automatically**
        - detect file roles
        - join compatible raw files
        - clean and standardize fields
        - score leads
        - generate explanations
        - return a downloadable scored file
        """
    )

if not uploaded_files:
    st.info("Upload one or more raw CSVs to begin.")
    st.stop()

st.subheader("File intake summary")
st.dataframe(prepare_upload_summary(uploaded_files), use_container_width=True, hide_index=True)

for file in uploaded_files:
    try:
        file.seek(0)
    except Exception:
        pass

file_frames: Dict[str, List[pd.DataFrame]] = {}
skipped_files: List[str] = []

for file in uploaded_files:
    try:
        df = pd.read_csv(file)
        if df.empty or len(df.columns) == 0:
            skipped_files.append(f"{file.name} (empty or no columns)")
            continue
    except Exception as exc:
        skipped_files.append(f"{file.name} ({exc})")
        continue

    role = detect_role(file.name)
    file_frames.setdefault(role, []).append(df)

if skipped_files:
    st.warning("Some files were skipped because they could not be parsed:")
    for item in skipped_files:
        st.write(f"- {item}")

if not file_frames:
    st.error("None of the uploaded files could be used.")
    st.stop()

preview_source = next(iter(file_frames.values()))[0]
st.subheader("Preview of first usable file")
st.dataframe(preview_source.head(10), use_container_width=True)

if not run_button:
    st.caption("Click **Score Leads** to run the automated pipeline.")
    st.stop()

try:
    merged_raw = build_primary_frame(file_frames)
except Exception as exc:
    st.error(f"Could not build the automated input frame: {exc}")
    st.stop()

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

scored_df = scored_df.loc[:, ~scored_df.columns.duplicated()].copy()
ranked = scored_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
ranked["display_name"] = ranked.apply(lead_label, axis=1)
ranked = ranked.drop_duplicates(
    subset=[c for c in ["lead_id", "lead_owner_id", "name", "company"] if c in ranked.columns],
    keep="first",
).reset_index(drop=True)

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

tab_overview, tab_top, tab_detail, tab_export = st.tabs(
    ["Overview", "Top leads", "Lead detail", "Export"]
)

with tab_overview:
    st.markdown("### What the system just did")
    st.write(
        "It accepted raw client files, standardized the schema, built the canonical lead table, scored the leads, and generated a business-friendly priority output."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Usable files", f"{sum(len(v) for v in file_frames.values())}")
    c2.metric("Detected roles", f"{len(file_frames)}")
    c3.metric("Output rows", f"{len(scored_df):,}")

    st.markdown("#### File roles detected")
    detected_rows = [{"file": f.name, "role": detect_role(f.name)} for f in uploaded_files]
    st.dataframe(pd.DataFrame(detected_rows), use_container_width=True, hide_index=True)

with tab_top:
    st.markdown("### Business view of the ranked leads")
    bucket_filter = st.selectbox("Filter by bucket", ["All", "Hot", "Warm", "Cool", "Cold"], index=0)
    search_text = st.text_input("Search by lead, company, owner, source, or status", value="")
    shortlist_n = (
        st.slider(
            "How many top leads to show",
            min_value=10,
            max_value=min(200, len(ranked)),
            value=min(25, len(ranked)),
            step=5,
        )
        if len(ranked) >= 10
        else len(ranked)
    )

    filtered = ranked.copy()
    if bucket_filter != "All" and "priority_bucket" in filtered.columns:
        filtered = filtered[filtered["priority_bucket"] == bucket_filter]

    if search_text.strip():
        q = search_text.strip().lower()
        search_cols = [
            c for c in ["display_name", "lead_owner_id", "lead_source", "status", "owner_role", "name", "company"]
            if c in filtered.columns
        ]
        mask = pd.Series(False, index=filtered.index)
        for col in search_cols:
            mask = mask | filtered[col].astype(str).str.lower().str.contains(q, na=False)
        filtered = filtered[mask]

    fixed_view_cols = [
        c
        for c in [
            "rank",
            "display_name",
            "lead_source",
            "status",
            "owner_role",
            "model_probability",
            "priority_score",
            "priority_bucket",
            "recommended_action",
        ]
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[fixed_view_cols].head(shortlist_n),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

with tab_detail:
    st.markdown("### Inspect one lead")
    lead_options = [
        f"{i + 1}. {ranked.loc[i, 'display_name']} | {ranked.loc[i, 'priority_bucket']} | {float(ranked.loc[i, 'priority_score']):.1f}"
        for i in range(len(ranked))
    ]
    selected_label = st.selectbox("Select a lead", lead_options, index=0)
    selected_idx = lead_options.index(selected_label)
    selected = ranked.loc[selected_idx]

    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown("#### Lead summary")
        summary_fields = [
            "display_name",
            "lead_id",
            "lead_owner_id",
            "lead_source",
            "status",
            "owner_role",
            "model_probability",
            "priority_score",
            "priority_bucket",
            "recommended_action",
        ]
        summary_rows = []
        for field in summary_fields:
            if field in selected.index:
                summary_rows.append(
                    {"Field": field.replace("_", " ").title(), "Value": display_value(selected[field])}
                )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### Explanation")
        st.write(f"**Bucket:** {selected.get('priority_bucket', 'Cold')}")
        st.write(f"**Recommended action:** {selected.get('recommended_action', 'Review')}")
        st.write(f"**Reason 1:** {selected.get('reason_1', '')}")
        st.write(f"**Reason 2:** {selected.get('reason_2', '')}")
        st.write(f"**Reason 3:** {selected.get('reason_3', '')}")
        if "xai_summary" in selected:
            st.info(selected.get("xai_summary", ""))

    with st.expander("Show full scored record"):
        clean_record = {}
        for k, v in selected.to_dict().items():
            if pd.isna(v):
                continue
            clean_record[k] = display_value(v)
        st.json(clean_record)

with tab_export:
    st.markdown("### Download the scored output")
    st.write("The exported CSV includes the ranked result set plus explanations and recommended actions.")
    st.download_button(
        label="Download scored CSV",
        data=scored_df.to_csv(index=False).encode("utf-8"),
        file_name="scored_leads.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "This dashboard supports one raw CSV or multiple raw CSVs. The client only uploads files and receives scored, explained output."
)