from pathlib import Path
import re

import pandas as pd


RAW_DIR = Path("data/raw")


def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def main() -> None:
    lead_path = RAW_DIR / "lead.csv"
    account_path = RAW_DIR / "accounts.csv"

    if not lead_path.exists():
        raise FileNotFoundError(f"Missing file: {lead_path}")
    if not account_path.exists():
        raise FileNotFoundError(f"Missing file: {account_path}")

    leads = pd.read_csv(lead_path)
    accounts = pd.read_csv(account_path)

    leads["company_norm"] = leads["company"].apply(normalize_text)
    accounts["account_name_norm"] = accounts["name"].apply(normalize_text)

    account_names = set(accounts["account_name_norm"].dropna())

    matches = leads["company_norm"].isin(account_names)

    print("=" * 80)
    print("ACCOUNT MATCH CHECK")
    print("=" * 80)
    print(f"Lead rows: {len(leads)}")
    print(f"Account rows: {len(accounts)}")
    print(f"Matched leads: {matches.sum()}")
    print(f"Unmatched leads: {(~matches).sum()}")
    print(f"Match rate: {matches.mean():.4f}")

    print("\nSample lead companies:")
    print(leads["company"].head(10).tolist())

    print("\nSample account names:")
    print(accounts["name"].head(10).tolist())


if __name__ == "__main__":
    main()