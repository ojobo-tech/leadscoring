from pathlib import Path
import re

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def standardize_booleans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                    }
                )
            )
    return df


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def clean_leads(lead_df: pd.DataFrame) -> pd.DataFrame:
    lead_df = lead_df.copy()

    lead_df = parse_dates(lead_df, ["created_date"])
    lead_df = standardize_booleans(lead_df, ["is_converted"])

    lead_df = lead_df.rename(
        columns={
            "id": "lead_id",
            "owner_id": "lead_owner_id",
            "created_date": "lead_created_date",
            "is_converted": "lead_is_converted",
        }
    )

    for col in ["first_name", "last_name", "company", "email", "phone", "status", "lead_source"]:
        if col in lead_df.columns:
            lead_df[col] = lead_df[col].astype(str).str.strip()

    lead_df["company_norm"] = lead_df["company"].apply(normalize_text)

    return lead_df


def clean_accounts(accounts_df: pd.DataFrame) -> pd.DataFrame:
    accounts_df = accounts_df.copy()

    accounts_df = parse_dates(accounts_df, ["created_date"])
    accounts_df = standardize_booleans(accounts_df, ["is_active"])

    accounts_df = accounts_df.rename(
        columns={
            "id": "account_id",
            "name": "account_name",
            "owner_id": "account_owner_id",
            "type": "account_type",
            "industry": "account_industry",
            "annual_revenue": "account_annual_revenue",
            "employee_count": "account_employee_count",
            "billing_city": "account_billing_city",
            "billing_state": "account_billing_state",
            "billing_country": "account_billing_country",
            "created_date": "account_created_date",
            "is_active": "account_is_active",
        }
    )

    accounts_df["account_name_norm"] = accounts_df["account_name"].apply(normalize_text)

    return accounts_df


def clean_contacts(contacts_df: pd.DataFrame) -> pd.DataFrame:
    contacts_df = contacts_df.copy()

    contacts_df = parse_dates(contacts_df, ["created_date"])
    contacts_df = standardize_booleans(contacts_df, ["is_primary"])

    contacts_df = contacts_df.rename(
        columns={
            "id": "contact_id",
            "account_id": "contact_account_id",
            "owner_id": "contact_owner_id",
            "first_name": "contact_first_name",
            "last_name": "contact_last_name",
            "email": "contact_email",
            "phone": "contact_phone",
            "title": "contact_title",
            "department": "contact_department",
            "lead_source": "contact_lead_source",
            "created_date": "contact_created_date",
            "is_primary": "contact_is_primary",
        }
    )

    return contacts_df


def clean_events(events_df: pd.DataFrame) -> pd.DataFrame:
    events_df = events_df.copy()

    events_df = parse_dates(events_df, ["start_datetime", "end_datetime", "created_date"])
    events_df = standardize_booleans(events_df, ["is_all_day_event"])

    events_df = events_df.rename(
        columns={
            "id": "event_id",
            "subject": "event_subject",
            "type": "event_type",
            "owner_id": "event_owner_id",
            "who_id": "event_who_id",
            "what_id": "event_what_id",
            "location": "event_location",
            "start_datetime": "event_start_datetime",
            "end_datetime": "event_end_datetime",
            "is_all_day_event": "event_is_all_day_event",
            "created_date": "event_created_date",
        }
    )

    return events_df


def build_task_features(tasks_df: pd.DataFrame) -> pd.DataFrame:
    tasks_df = tasks_df.copy()
    tasks_df = standardize_booleans(tasks_df, ["is_closed"])
    tasks_df = parse_dates(tasks_df, ["activity_date", "created_date"])

    lead_tasks = tasks_df.dropna(subset=["who_id"]).copy()

    agg = (
        lead_tasks.groupby("who_id")
        .agg(
            task_count=("id", "count"),
            open_task_count=("is_closed", lambda x: (~x.fillna(False)).sum()),
            closed_task_count=("is_closed", "sum"),
            high_priority_task_count=("priority", lambda x: (x.astype(str).str.lower() == "high").sum()),
            last_task_created_date=("created_date", "max"),
            last_task_activity_date=("activity_date", "max"),
        )
        .reset_index()
        .rename(columns={"who_id": "lead_id"})
    )

    return agg


def build_user_context(users_df: pd.DataFrame) -> pd.DataFrame:
    users_df = users_df.copy()
    users_df = standardize_booleans(users_df, ["is_active"])
    users_df = parse_dates(users_df, ["hire_date"])

    users_df = users_df.rename(
        columns={
            "id": "user_id",
            "first_name": "owner_first_name",
            "last_name": "owner_last_name",
            "name": "owner_name",
            "title": "owner_title",
            "role": "owner_role",
            "department": "owner_department",
            "is_active": "owner_is_active",
            "hire_date": "owner_hire_date",
        }
    )

    return users_df[
        [
            "user_id",
            "owner_first_name",
            "owner_last_name",
            "owner_name",
            "owner_title",
            "owner_role",
            "owner_department",
            "owner_is_active",
            "owner_hire_date",
        ]
    ]


def build_account_contact_features(accounts_df: pd.DataFrame, contacts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build account-level enrichment and contact aggregates.
    """
    accounts_df = clean_accounts(accounts_df)
    contacts_df = clean_contacts(contacts_df)

    contact_agg = (
        contacts_df.groupby("contact_account_id")
        .agg(
            contact_count=("contact_id", "count"),
            primary_contact_count=("contact_is_primary", "sum"),
            unique_contact_departments=("contact_department", pd.Series.nunique),
            unique_contact_titles=("contact_title", pd.Series.nunique),
            lead_source_contact_mode=("contact_lead_source", lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA),
            last_contact_created_date=("contact_created_date", "max"),
        )
        .reset_index()
        .rename(columns={"contact_account_id": "account_id"})
    )

    account_context = accounts_df.merge(
        contact_agg,
        how="left",
        on="account_id",
    )

    return account_context


def build_event_features(events_df: pd.DataFrame, contacts_df: pd.DataFrame, accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create account-level event activity signals from:
    - event.what_id when it points to an account
    - event.who_id via contact -> account linkage
    """
    events_df = clean_events(events_df)
    contacts_df = clean_contacts(contacts_df)
    accounts_df = clean_accounts(accounts_df)

    # Event signals tied directly to accounts via what_id
    account_events_direct = events_df[
        events_df["event_what_id"].astype(str).str.startswith("001", na=False)
    ].copy()

    direct_agg = (
        account_events_direct.groupby("event_what_id")
        .agg(
            account_event_count=("event_id", "count"),
            account_demo_count=("event_type", lambda x: (x.astype(str).str.lower() == "demo").sum()),
            account_meeting_count=("event_type", lambda x: (x.astype(str).str.lower() == "meeting").sum()),
            account_onsite_count=("event_type", lambda x: (x.astype(str).str.lower() == "onsite visit").sum()),
            account_qbr_count=("event_type", lambda x: (x.astype(str).str.lower() == "qbr").sum()),
            last_account_event_date=("event_created_date", "max"),
        )
        .reset_index()
        .rename(columns={"event_what_id": "account_id"})
    )

    # Event signals tied to contacts, then rolled up to accounts
    contact_events = events_df.merge(
        contacts_df[["contact_id", "contact_account_id"]],
        how="left",
        left_on="event_who_id",
        right_on="contact_id",
    )

    contact_agg = (
        contact_events.dropna(subset=["contact_account_id"])
        .groupby("contact_account_id")
        .agg(
            contact_event_count=("event_id", "count"),
            contact_demo_count=("event_type", lambda x: (x.astype(str).str.lower() == "demo").sum()),
            contact_meeting_count=("event_type", lambda x: (x.astype(str).str.lower() == "meeting").sum()),
            contact_qbr_count=("event_type", lambda x: (x.astype(str).str.lower() == "qbr").sum()),
            last_contact_event_date=("event_created_date", "max"),
        )
        .reset_index()
        .rename(columns={"contact_account_id": "account_id"})
    )

    event_features = accounts_df[["account_id"]].drop_duplicates()

    event_features = event_features.merge(direct_agg, how="left", on="account_id")
    event_features = event_features.merge(contact_agg, how="left", on="account_id")

    # Fill zero-like counts
    for col in [
        "account_event_count",
        "account_demo_count",
        "account_meeting_count",
        "account_onsite_count",
        "account_qbr_count",
        "contact_event_count",
        "contact_demo_count",
        "contact_meeting_count",
        "contact_qbr_count",
    ]:
        if col in event_features.columns:
            event_features[col] = event_features[col].fillna(0).astype(int)

    return event_features


def build_lead_master_v2() -> pd.DataFrame:
    lead_df = load_csv("lead.csv")
    accounts_df = load_csv("accounts.csv")
    contacts_df = load_csv("contact.csv")
    tasks_df = load_csv("tasks.csv")
    users_df = load_csv("user.csv")
    events_df = load_csv("event.csv")

    lead_df = clean_leads(lead_df)
    task_features = build_task_features(tasks_df)
    user_context = build_user_context(users_df)
    account_context = build_account_contact_features(accounts_df, contacts_df)
    event_features = build_event_features(events_df, contacts_df, accounts_df)

    # Join user context
    lead_master = lead_df.merge(
        user_context,
        how="left",
        left_on="lead_owner_id",
        right_on="user_id",
    )

    # Join task features
    lead_master = lead_master.merge(
        task_features,
        how="left",
        on="lead_id",
    )

    # Match lead company to account name (approximate, based on normalized text)
    lead_master = lead_master.merge(
        account_context,
        how="left",
        left_on="company_norm",
        right_on="account_name_norm",
        suffixes=("", "_acct"),
    )

    # Join event features on matched account
    lead_master = lead_master.merge(
        event_features,
        how="left",
        on="account_id",
    )

    # Fill missing counts with zeros
    count_cols = [
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
        "contact_count",
        "primary_contact_count",
        "account_event_count",
        "account_demo_count",
        "account_meeting_count",
        "account_onsite_count",
        "account_qbr_count",
        "contact_event_count",
        "contact_demo_count",
        "contact_meeting_count",
        "contact_qbr_count",
    ]
    for col in count_cols:
        if col in lead_master.columns:
            lead_master[col] = lead_master[col].fillna(0).astype(int)

    # Convert dates to useful recency features
    if "lead_created_date" in lead_master.columns:
        lead_master["lead_created_date"] = pd.to_datetime(lead_master["lead_created_date"], errors="coerce")
        reference_date = lead_master["lead_created_date"].max()
        lead_master["lead_age_days"] = (reference_date - lead_master["lead_created_date"]).dt.days

    if "owner_hire_date" in lead_master.columns and "lead_created_date" in lead_master.columns:
        lead_master["owner_tenure_days"] = (
            lead_master["lead_created_date"] - pd.to_datetime(lead_master["owner_hire_date"], errors="coerce")
        ).dt.days

    if "last_task_created_date" in lead_master.columns and "lead_created_date" in lead_master.columns:
        lead_master["days_since_last_task_created"] = (
            lead_master["lead_created_date"] - pd.to_datetime(lead_master["last_task_created_date"], errors="coerce")
        ).dt.days

    if "last_task_activity_date" in lead_master.columns and "lead_created_date" in lead_master.columns:
        lead_master["days_since_last_task_activity"] = (
            lead_master["lead_created_date"] - pd.to_datetime(lead_master["last_task_activity_date"], errors="coerce")
        ).dt.days

    if "last_contact_created_date" in lead_master.columns and "lead_created_date" in lead_master.columns:
        lead_master["days_since_last_contact_created"] = (
            lead_master["lead_created_date"] - pd.to_datetime(lead_master["last_contact_created_date"], errors="coerce")
        ).dt.days

    if "last_account_event_date" in lead_master.columns and "lead_created_date" in lead_master.columns:
        lead_master["days_since_last_account_event"] = (
            lead_master["lead_created_date"] - pd.to_datetime(lead_master["last_account_event_date"], errors="coerce")
        ).dt.days

    # Standardize booleans to integers for modeling
    for col in ["lead_is_converted", "owner_is_active", "account_is_active"]:
        if col in lead_master.columns:
            lead_master[col] = lead_master[col].astype(str).str.lower().map(
                {
                    "true": 1,
                    "false": 0,
                    "1": 1,
                    "0": 0,
                    "yes": 1,
                    "no": 0,
                }
            )

    # Drop leakage-prone and unnecessary identifier fields
    drop_cols = [
        "first_name",
        "last_name",
        "email",
        "phone",
        "converted_account_id",
        "converted_contact_id",
        "converted_opportunity_id",
        "owner_name",
        "owner_title",
        "owner_first_name",
        "owner_last_name",
        "lead_created_date",
        "last_task_created_date",
        "last_task_activity_date",
        "owner_hire_date",
        "account_name",
        "account_name_norm",
        "contact_lead_source_mode",
        "last_contact_created_date",
        "last_account_event_date",
        "lead_company",
    ]
    # keep lead_id for traceability only if you want; drop it for modeling later
    # for master file, it's okay to keep it
    lead_master = lead_master.drop(columns=[c for c in drop_cols if c in lead_master.columns], errors="ignore")

    # Put target at the end
    target_col = "lead_is_converted"
    cols = [c for c in lead_master.columns if c != target_col] + [target_col]
    lead_master = lead_master[cols]

    return lead_master


def main() -> None:
    df = build_lead_master_v2()
    output_path = PROCESSED_DIR / "lead_master_v2.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved processed file to: {output_path}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(list(df.columns))


if __name__ == "__main__":
    main()