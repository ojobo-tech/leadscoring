from pathlib import Path
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
            df[col] = df[col].astype(str).str.lower().map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False,
                }
            )
    return df


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_leads(lead_df: pd.DataFrame) -> pd.DataFrame:
    lead_df = lead_df.copy()

    # Standardize key fields
    lead_df = parse_dates(lead_df, ["created_date"])
    lead_df = standardize_booleans(lead_df, ["is_converted"])

    # Keep IDs and target, but drop leakage-prone converted IDs from features later
    lead_df = lead_df.rename(
        columns={
            "id": "lead_id",
            "owner_id": "lead_owner_id",
            "created_date": "lead_created_date",
            "is_converted": "lead_is_converted",
        }
    )

    # Basic text cleanup
    for col in ["first_name", "last_name", "company", "email", "phone", "status", "lead_source"]:
        if col in lead_df.columns:
            lead_df[col] = lead_df[col].astype(str).str.strip()

    return lead_df


def build_task_features(tasks_df: pd.DataFrame) -> pd.DataFrame:
    tasks_df = tasks_df.copy()
    tasks_df = standardize_booleans(tasks_df, ["is_closed"])
    tasks_df = parse_dates(tasks_df, ["activity_date", "created_date"])

    # We only use tasks linked directly to a lead through who_id
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
            "owner_name",
            "owner_title",
            "owner_role",
            "owner_department",
            "owner_is_active",
            "owner_hire_date",
        ]
    ]


def build_lead_master() -> pd.DataFrame:
    lead_df = load_csv("lead.csv")
    tasks_df = load_csv("tasks.csv")
    users_df = load_csv("user.csv")

    lead_df = clean_leads(lead_df)
    task_features = build_task_features(tasks_df)
    user_context = build_user_context(users_df)

    # Merge owner context
    lead_master = lead_df.merge(
        user_context,
        how="left",
        left_on="lead_owner_id",
        right_on="user_id",
    )

    # Merge lead-level task activity
    lead_master = lead_master.merge(
        task_features,
        how="left",
        on="lead_id",
    )

    # Fill missing task counts with zeros
    for col in [
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
    ]:
        if col in lead_master.columns:
            lead_master[col] = lead_master[col].fillna(0).astype(int)

    # Remove duplicate user_id helper column after merge
    if "user_id" in lead_master.columns:
        lead_master = lead_master.drop(columns=["user_id"])

    # Optional: create a simple recency feature
    if "lead_created_date" in lead_master.columns:
        reference_date = pd.to_datetime(lead_master["lead_created_date"]).max()
        lead_master["lead_age_days"] = (reference_date - lead_master["lead_created_date"]).dt.days

    # Keep the target at the end
    target_col = "lead_is_converted"
    cols = [c for c in lead_master.columns if c != target_col] + [target_col]
    lead_master = lead_master[cols]

    return lead_master


def main() -> None:
    lead_master = build_lead_master()

    output_path = PROCESSED_DIR / "lead_master.csv"
    lead_master.to_csv(output_path, index=False)

    print(f"Saved processed file to: {output_path}")
    print(f"Shape: {lead_master.shape}")
    print("Columns:")
    print(list(lead_master.columns))


if __name__ == "__main__":
    main()