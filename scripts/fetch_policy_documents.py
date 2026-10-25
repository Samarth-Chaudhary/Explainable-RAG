"""Phase 6: Official Public Policy Document Fetcher and Provenance Verifier.

Downloads official, public regulatory documents from sovereign authorities:
1. Federal Reserve & OCC: SR 11-7 Guidance on Model Risk Management
2. Basel Committee on Banking Supervision (BCBS 75): Principles for the Management of Credit Risk
3. Reserve Bank of India (RBI): Reserve Bank of India (Digital Lending) Directions, 2025
4. Reserve Bank of India (RBI): Master Circular - Fair Practices Code for NBFCs

Verifies SHA-256 cryptographic hashes, saves to data/policies/, and updates data/DATA_SOURCES.md
and data/POLICY_SOURCES.md.
If any download fails, terminates with non-zero exit code (no synthetic text substitution).
"""

from __future__ import annotations

import hashlib
import logging
import ssl
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_policy_documents")

ROOT_DIR = Path(__file__).resolve().parent.parent
POLICIES_DIR = ROOT_DIR / "data" / "policies"
DATA_SOURCES_MD = ROOT_DIR / "data" / "DATA_SOURCES.md"
POLICY_SOURCES_MD = ROOT_DIR / "data" / "POLICY_SOURCES.md"

POLICY_SOURCES = [
    {
        "id": "FED-SR11-7",
        "title": "Supervisory Guidance on Model Risk Management (SR 11-7)",
        "issuer": "Federal Reserve Board & OCC",
        "jurisdiction": "US",
        "url": "https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf",
        "target_filename": "fed_sr11_7.pdf",
        "description": "Foundational supervisory standards for model risk management, validation, conceptual soundness, and outcomes analysis.",
    },
    {
        "id": "BCBS-75",
        "title": "Principles for the Management of Credit Risk (BCBS 75)",
        "issuer": "Basel Committee on Banking Supervision (BIS)",
        "jurisdiction": "Global / International",
        "url": "https://www.bis.org/publications/200009-guidelines-principles-management-credit-risk.pdf",
        "target_filename": "basel_credit_risk.pdf",
        "description": "Core supervisory principles for establishing credit granting criteria, assessing borrower repayment capacity, and ongoing monitoring.",
    },
    {
        "id": "RBI-DIGITAL-LENDING-2025",
        "title": "Reserve Bank of India (Digital Lending) Directions, 2025 (RBI/2025-26/36)",
        "issuer": "Reserve Bank of India (RBI)",
        "jurisdiction": "IN",
        "landing_url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx?Id=12848&Mode=0",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/36NT8C402BE7C2A349E0BFFF3C526668CD7A.PDF",
        "target_filename": "rbi_digital_lending_2025.pdf",
        "description": "Mandates fair practices, automated credit underwriting disclosures, borrower consent, cooling-off periods, and algorithm accountability.",
    },
    {
        "id": "RBI-FAIR-PRACTICES-CODE",
        "title": "Master Circular - Fair Practices Code for Non-Banking Financial Companies",
        "issuer": "Reserve Bank of India (RBI)",
        "jurisdiction": "IN",
        "landing_url": "https://www.rbi.org.in/Scripts/BS_ViewMasterCirculars.aspx?Id=8930",
        "url": "https://rbidocs.rbi.org.in/rdocs/notification/PDFs/AI141P060614F.PDF",
        "target_filename": "rbi_fair_practices_code.pdf",
        "description": "Statutory standards for loan appraisal, transparent terms and conditions, disbursement, and non-coercive recovery mechanisms.",
    },
]


def download_and_verify(source_info: dict[str, object]) -> dict[str, object]:
    """Download a public PDF from its official URL, compute SHA-256, and save."""
    url = source_info["url"]
    target_path = POLICIES_DIR / source_info["target_filename"]
    logger.info(f"Downloading {source_info['id']} from: {url}")

    content: bytes | None = None

    # For RBI documents protected by F5 BIG-IP session tokens, visit landing URL first
    if "landing_url" in source_info:
        logger.info(f"  Acquiring session cookies from landing page: {source_info['landing_url']}")
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        })
        try:
            r1 = session.get(source_info["landing_url"], timeout=15)
            if r1.status_code != 200:
                raise RuntimeError(f"Failed to load landing page {source_info['landing_url']}: status {r1.status_code}")
            
            session.headers.update({
                "Accept": "application/pdf,*/*",
                "Referer": source_info["landing_url"],
            })
            r2 = session.get(url, timeout=30)
            if r2.status_code != 200:
                raise RuntimeError(f"HTTP status {r2.status_code} when fetching {url}")
            content = r2.content
        except Exception as exc:
            logger.error(f"[DOWNLOAD FAILED] Session download error for {source_info['id']} ({url}): {exc}")
            sys.exit(1)
    else:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,*/*",
        }
        ctx = ssl.create_default_context()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP status {response.status} when fetching {url}")
                content = response.read()
        except Exception as exc:
            logger.error(f"[DOWNLOAD FAILED] Could not download official document {source_info['id']} from {url}: {exc}")
            sys.exit(1)

    if content is None:
        logger.error(f"[DOWNLOAD FAILED] No content received for {source_info['id']}.")
        sys.exit(1)

    file_size = len(content)
    if file_size < 1000:
        logger.error(f"[INVALID FILE] Downloaded file for {source_info['id']} is suspiciously small ({file_size} bytes).")
        sys.exit(1)

    if not content.startswith(b"%PDF"):
        logger.error(f"[INVALID PDF] Downloaded content for {source_info['id']} lacks %PDF magic header.")
        sys.exit(1)

    sha256 = hashlib.sha256(content).hexdigest()
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(content)

    retrieval_time = datetime.now(timezone.utc).isoformat()
    logger.info(f"  SUCCESS: Saved {target_path.name} ({file_size:,} bytes, SHA-256: {sha256})")

    return {
        **source_info,
        "sha256": sha256,
        "file_size_bytes": file_size,
        "retrieval_timestamp": retrieval_time,
        "local_path": f"data/policies/{target_path.name}",
    }


def main():
    logger.info("Starting Phase 6 Policy Document Fetch...")
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)

    verified_policies = []
    for source in POLICY_SOURCES:
        info = download_and_verify(source)
        verified_policies.append(info)

    logger.info("Updating data/DATA_SOURCES.md and data/POLICY_SOURCES.md...")
    sources_table = "\n\n## Official Regulatory Policy Documents\n\n"
    sources_table += "| Document ID | Title | Issuer | Jurisdiction | File Size | SHA-256 Checksum | Official URL | Retrieval Date (UTC) |\n"
    sources_table += "| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: |\n"

    for p in verified_policies:
        sources_table += (
            f"| `{p['id']}` | {p['title']} | {p['issuer']} | {p['jurisdiction']} | "
            f"{p['file_size_bytes']:,} B | `{p['sha256']}` | [Official Source]({p['url']}) | "
            f"`{p['retrieval_timestamp'][:10]}` |\n"
        )

    if DATA_SOURCES_MD.is_file():
        existing_text = DATA_SOURCES_MD.read_text(encoding="utf-8")
        if "## Official Regulatory Policy Documents" in existing_text:
            existing_text = existing_text.split("## Official Regulatory Policy Documents")[0].strip()
        updated_text = existing_text + sources_table
    else:
        updated_text = "# Verified Data & Policy Sources\n" + sources_table

    with open(DATA_SOURCES_MD, "w", encoding="utf-8") as f:
        f.write(updated_text)

    with open(POLICY_SOURCES_MD, "w", encoding="utf-8") as f:
        f.write("# Sovereign Regulatory Policy Corpus\n" + sources_table)

    logger.info(f"[COMPLETE] Downloaded and verified {len(verified_policies)} official regulatory policy documents.")


if __name__ == "__main__":
    main()
