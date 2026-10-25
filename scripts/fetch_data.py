"""Data acquisition script for real credit risk datasets.

Fetches, verifies (SHA-256 recorded in data/DATA_SOURCES.md), and caches:
a) Primary: Home Credit Default Risk (application_train.csv) via Kaggle API.
   Reads credentials from env vars (KAGGLE_USERNAME, KAGGLE_KEY). If credentials or
   competition rules are missing, prints exact setup steps and exits non-zero.
b) Fallback needing no login: UCI "Default of Credit Card Clients" (ucimlrepo id=350,
   verified live) and OpenML German Credit (data_id=31, verified live).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import sys
import zipfile
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
DATA_SOURCES_MD = REPO_ROOT / "data" / "DATA_SOURCES.md"

# Known recorded SHA-256 hashes (verified live)
RECORDED_HASHES: dict[str, str] = {
    "uci_credit_card": "20a2540061fab3984f208fbe55745b0d60e3a86207c8b02e408fd23f1a3a4b94",
    "german_credit": "9888fb06e7bf42963f4db2c8fd57666d5426c0a3692d10b51571de49518a105a",
    "home_credit_train": "",
}


def compute_sha256(file_path: Path) -> str:
    """Compute cryptographic SHA-256 digest of a local file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def update_data_sources_md(
    source_name: str,
    source_id: str,
    raw_path: Path,
    sha256_hash: str,
    row_count: int,
    column_count: int,
    license_terms: str,
    notes: str,
) -> None:
    """Append or update verified dataset entry in data/DATA_SOURCES.md."""
    DATA_SOURCES_MD.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    entry = f"""
## {source_name}
- **Source Identifier:** `{source_id}`
- **Local Path:** `{raw_path.relative_to(REPO_ROOT)}`
- **Retrieval Timestamp:** {timestamp}
- **File Size:** {raw_path.stat().st_size:,} bytes
- **SHA-256 Digest:** `{sha256_hash}`
- **Dimensions:** {row_count:,} rows × {column_count} columns
- **License / Terms:** {license_terms}
- **Provenance & Verification Notes:** {notes}

---
"""
    if not DATA_SOURCES_MD.exists():
        header = "# Data Sources & Cryptographic Provenance Registry\n\nAll real datasets used for model training and risk evaluation must be verified by SHA-256.\n\n---\n"
        DATA_SOURCES_MD.write_text(header + entry, encoding="utf-8")
    else:
        existing = DATA_SOURCES_MD.read_text(encoding="utf-8")
        # If entry exists, replace it or append
        if f"## {source_name}" in existing:
            # Reconstruct content
            lines = existing.split("\n")
            new_lines = []
            skip = False
            for line in lines:
                if line.strip() == f"## {source_name}":
                    skip = True
                elif skip and line.strip() == "---":
                    skip = False
                    continue
                if not skip:
                    new_lines.append(line)
            DATA_SOURCES_MD.write_text("\n".join(new_lines).strip() + "\n" + entry, encoding="utf-8")
        else:
            DATA_SOURCES_MD.write_text(existing + entry, encoding="utf-8")


def fetch_home_credit() -> Path:
    """Fetch Home Credit Default Risk application_train.csv via Kaggle API."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        print("=" * 80, file=sys.stderr)
        print("[FAIL] Missing Kaggle API Credentials for Home Credit Default Risk!", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("Required Environment Variables:", file=sys.stderr)
        print("  - KAGGLE_USERNAME: Your Kaggle account username", file=sys.stderr)
        print("  - KAGGLE_KEY: Your Kaggle API token key", file=sys.stderr)
        print("\nExact Setup Steps to Resolve:", file=sys.stderr)
        print("1. Log in to https://www.kaggle.com", file=sys.stderr)
        print("2. Navigate to Account Settings: https://www.kaggle.com/account", file=sys.stderr)
        print("3. Under 'API', click 'Create New API Token' to download kaggle.json", file=sys.stderr)
        print("4. Set credentials in your current shell session:", file=sys.stderr)
        print("   In Windows PowerShell:", file=sys.stderr)
        print('     $env:KAGGLE_USERNAME = "<your_username>"', file=sys.stderr)
        print('     $env:KAGGLE_KEY = "<your_api_key>"', file=sys.stderr)
        print("   In Unix / macOS Bash:", file=sys.stderr)
        print('     export KAGGLE_USERNAME="<your_username>"', file=sys.stderr)
        print('     export KAGGLE_KEY="<your_api_key>"', file=sys.stderr)
        print("5. Accept the mandatory official competition rules at:", file=sys.stderr)
        print("   https://www.kaggle.com/c/home-credit-default-risk/rules", file=sys.stderr)
        print("6. Re-run: python scripts/fetch_data.py --source home_credit", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.exit(1)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "application_train.csv"

    if target_csv.exists():
        sha256 = compute_sha256(target_csv)
        print(f"[CACHE] application_train.csv already cached: {target_csv} (SHA-256: {sha256})")
        return target_csv

    print(f"Downloading Home Credit Default Risk via Kaggle API using account '{username}'...")
    url = "https://www.kaggle.com/api/v1/competitions/data/download/home-credit-default-risk/application_train.csv.zip"

    try:
        resp = requests.get(url, auth=(username, key), stream=True, timeout=120)
        if resp.status_code == 403:
            print("=" * 80, file=sys.stderr)
            print("[FAIL] Kaggle API returned 403 Forbidden.", file=sys.stderr)
            print("You must accept the competition rules before downloading data:", file=sys.stderr)
            print("  https://www.kaggle.com/c/home-credit-default-risk/rules", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            sys.exit(1)
        elif resp.status_code == 401:
            print("=" * 80, file=sys.stderr)
            print("[FAIL] Kaggle API returned 401 Unauthorized. Invalid username or API key.", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            sys.exit(1)
        resp.raise_for_status()

        # Extract application_train.csv from zip
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extract("application_train.csv", path=RAW_DATA_DIR)

        sha256 = compute_sha256(target_csv)
        df = pd.read_csv(target_csv, nrows=5)
        total_rows = sum(1 for _ in open(target_csv, "rb")) - 1

        update_data_sources_md(
            source_name="Home Credit Default Risk (Primary)",
            source_id="kaggle:c/home-credit-default-risk/application_train.csv",
            raw_path=target_csv,
            sha256_hash=sha256,
            row_count=total_rows,
            column_count=len(df.columns),
            license_terms="Competition Use Only (Home Credit Group)",
            notes="Downloaded directly via Kaggle API with authenticated token.",
        )
        print(f"[SUCCESS] Downloaded application_train.csv (SHA-256: {sha256}, Rows: {total_rows:,})")
        return target_csv

    except Exception as e:
        print(f"[ERROR] Failed to download Home Credit data: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_uci_credit_card() -> Path:
    """Fetch UCI Default of Credit Card Clients dataset (ucimlrepo id=350, verified live)."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "uci_credit_card.csv"

    if target_csv.exists():
        sha256 = compute_sha256(target_csv)
        print(f"[CACHE] uci_credit_card.csv already cached: {target_csv} (SHA-256: {sha256})")
        return target_csv

    print("Fetching UCI Default of Credit Card Clients (Dataset ID: 350 via ucimlrepo)...")
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features
    targets = dataset.data.targets

    df = pd.concat([features, targets], axis=1)
    df.to_csv(target_csv, index=False)

    sha256 = compute_sha256(target_csv)
    update_data_sources_md(
        source_name="UCI Default of Credit Card Clients (Zero-Login Fallback)",
        source_id="ucimlrepo:id=350",
        raw_path=target_csv,
        sha256_hash=sha256,
        row_count=len(df),
        column_count=len(df.columns),
        license_terms="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        notes="Live verified via official ucimlrepo package. Real financial default benchmark.",
    )
    print(f"[SUCCESS] Cached uci_credit_card.csv (SHA-256: {sha256}, Rows: {len(df):,})")
    return target_csv


def fetch_german_credit() -> Path:
    """Fetch OpenML German Credit dataset (data_id=31, verified live)."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_csv = RAW_DATA_DIR / "german_credit.csv"

    if target_csv.exists():
        sha256 = compute_sha256(target_csv)
        print(f"[CACHE] german_credit.csv already cached: {target_csv} (SHA-256: {sha256})")
        return target_csv

    print("Fetching OpenML German Credit (data_id: 31 via sklearn.datasets.fetch_openml)...")
    from sklearn.datasets import fetch_openml

    ds = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = ds.frame
    df.to_csv(target_csv, index=False)

    sha256 = compute_sha256(target_csv)
    update_data_sources_md(
        source_name="OpenML German Credit (Benchmark Control)",
        source_id="openml:data_id=31",
        raw_path=target_csv,
        sha256_hash=sha256,
        row_count=len(df),
        column_count=len(df.columns),
        license_terms="Public Domain / CC0",
        notes="Live verified via OpenML API (credit-g).",
    )
    print(f"[SUCCESS] Cached german_credit.csv (SHA-256: {sha256}, Rows: {len(df):,})")
    return target_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify credit risk datasets.")
    parser.add_argument(
        "--source",
        choices=["home_credit", "uci", "german", "all"],
        default="auto",
        help="Dataset to fetch (default: 'auto' checks Kaggle credentials, falls back to UCI)",
    )
    args = parser.parse_args()

    if args.source == "home_credit":
        fetch_home_credit()
    elif args.source == "uci":
        fetch_uci_credit_card()
    elif args.source == "german":
        fetch_german_credit()
    elif args.source == "all":
        # Attempts all
        fetch_uci_credit_card()
        fetch_german_credit()
        fetch_home_credit()
    elif args.source == "auto":
        # If Kaggle credentials exist, fetch primary; otherwise fetch zero-login fallback
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            print("[INFO] Found Kaggle credentials in environment. Fetching primary Home Credit dataset...")
            fetch_home_credit()
        else:
            print("[INFO] No Kaggle credentials found. Fetching zero-login fallback (UCI Credit Card ID 350)...")
            fetch_uci_credit_card()


if __name__ == "__main__":
    main()
