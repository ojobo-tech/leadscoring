from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
FILES = [
    "lead.csv",
    "account.csv",
    "contact.csv",
    "event.csv",
    "opportunity.csv",
    "order.csv",
    "order_items.csv",
    "tasks.csv",
    "user.csv",
]


def inspect_file(file_path: Path) -> None:
    """Print basic profiling information for one CSV file."""
    df = pd.read_csv(file_path)

    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nFirst 5 rows:")
    print(df.head())


def main() -> None:
    for file_name in FILES:
        file_path = RAW_DIR / file_name

        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue

        inspect_file(file_path)


if __name__ == "__main__":
    main()