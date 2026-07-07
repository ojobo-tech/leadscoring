from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/eda_v3")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "lead_model_ready_v3.csv"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE)


def basic_info(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("BASIC INFO")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDtypes:")
    print(df.dtypes)


def target_balance(df: pd.DataFrame) -> None:
    if "lead_is_converted" not in df.columns:
        print("\nTarget column lead_is_converted not found.")
        return

    print("\n" + "=" * 80)
    print("TARGET BALANCE")
    print("=" * 80)

    print("\nValue counts:")
    print(df["lead_is_converted"].value_counts(dropna=False))

    print("\nValue proportions:")
    print(df["lead_is_converted"].value_counts(normalize=True, dropna=False))

    plt.figure(figsize=(6, 4))
    df["lead_is_converted"].value_counts().plot(kind="bar")
    plt.title("Target Distribution: lead_is_converted")
    plt.xlabel("Converted")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "target_distribution.png")
    plt.close()


def missing_values(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)

    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)

    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": missing_pct,
        }
    )

    print(missing_report.head(25))
    missing_report.to_csv(REPORTS_DIR / "missing_values_report.csv")


def numeric_summary(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        print("\nNo numeric columns found.")
        return

    print("\n" + "=" * 80)
    print("NUMERIC SUMMARY")
    print("=" * 80)

    summary = df[numeric_cols].describe().T
    print(summary)
    summary.to_csv(REPORTS_DIR / "numeric_summary.csv")

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        corr.to_csv(REPORTS_DIR / "correlation_matrix.csv")

        plt.figure(figsize=(12, 10))
        plt.imshow(corr, interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90, fontsize=7)
        plt.yticks(range(len(numeric_cols)), numeric_cols, fontsize=7)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "correlation_matrix.png")
        plt.close()


def categorical_review(df: pd.DataFrame, cols: list[str]) -> None:
    print("\n" + "=" * 80)
    print("CATEGORICAL REVIEW")
    print("=" * 80)

    for col in cols:
        if col in df.columns:
            print(f"\nColumn: {col}")
            print(df[col].astype(str).value_counts(dropna=False).head(20))


def feature_checks(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("FEATURE CHECKS")
    print("=" * 80)

    candidate_cols = [
        "status",
        "lead_source",
        "lead_owner_id",
        "owner_role",
        "owner_department",
        "owner_is_active",
        "task_count",
        "open_task_count",
        "closed_task_count",
        "high_priority_task_count",
        "lead_age_days",
        "owner_tenure_days",
        "days_since_last_task_created",
        "days_since_last_task_activity",
    ]

    existing_cols = [c for c in candidate_cols if c in df.columns]
    print("Columns available for review:")
    print(existing_cols)

    categorical_review(
        df,
        [
            "status",
            "lead_source",
            "lead_owner_id",
            "owner_role",
            "owner_department",
        ],
    )


def save_summary_outputs(df: pd.DataFrame) -> None:
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().sum() / len(df) * 100).values,
        }
    )
    profile.to_csv(REPORTS_DIR / "column_profile.csv", index=False)


def main() -> None:
    df = load_data()

    basic_info(df)
    target_balance(df)
    missing_values(df)
    numeric_summary(df)
    feature_checks(df)
    save_summary_outputs(df)

    print(f"\nEDA outputs saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()