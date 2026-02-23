"""
download_pdfs.py
----------------
Downloads strategic plan PDFs for all universities in metadata.csv.

Usage:
    python src/download_pdfs.py

PDFs are saved to:  data/raw_pdfs/<SafeUniversityName>.pdf
Skips files that are already downloaded.
"""

import os
import re
import time
import pandas as pd
import requests
from pathlib import Path

METADATA_PATH = Path(__file__).parent.parent / "data" / "metadata.csv"
PDF_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def safe_filename(name: str) -> str:
    """Convert a university name to a safe filename."""
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


def download_pdf(url: str, dest: Path, timeout: int = 30) -> bool:
    """
    Download a single PDF from url to dest.
    Returns True on success, False on failure.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            print(f"  [WARN] URL may not be a direct PDF (content-type: {content_type})")

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = dest.stat().st_size / 1024
        print(f"  [OK] {dest.name} ({size_kb:.0f} KB)")
        return True

    except requests.exceptions.HTTPError as e:
        print(f"  [FAIL] HTTP {e.response.status_code} — {url}")
    except requests.exceptions.ConnectionError:
        print(f"  [FAIL] Connection error — {url}")
    except requests.exceptions.Timeout:
        print(f"  [FAIL] Timeout — {url}")
    except Exception as e:
        print(f"  [FAIL] {e} — {url}")
    return False


def main():
    df = pd.read_csv(METADATA_PATH)
    print(f"Corpus: {len(df)} universities\n")

    results = {"downloaded": [], "skipped": [], "failed": []}

    for _, row in df.iterrows():
        name = row["university"]
        url = row["pdf_url"]
        safe_name = safe_filename(name)
        dest = PDF_DIR / f"{safe_name}.pdf"

        print(f"[{row['tier']:6s}] {name}")

        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  [SKIP] Already downloaded ({dest.stat().st_size // 1024} KB)")
            results["skipped"].append(name)
            continue

        # Skip placeholder / non-PDF URLs (HTML pages)
        if not url.endswith(".pdf") and "pdf" not in url.lower():
            print(f"  [SKIP] URL is likely an HTML page, not a direct PDF — manual download needed")
            print(f"         URL: {url}")
            results["failed"].append(name)
            continue

        success = download_pdf(url, dest)
        if success:
            results["downloaded"].append(name)
        else:
            results["failed"].append(name)

        time.sleep(1)  # polite crawl delay

    print("\n--- Summary ---")
    print(f"Downloaded : {len(results['downloaded'])}")
    print(f"Skipped    : {len(results['skipped'])}  (already on disk)")
    print(f"Failed     : {len(results['failed'])}")
    if results["failed"]:
        print("\nManual download required for:")
        for n in results["failed"]:
            row = df[df["university"] == n].iloc[0]
            print(f"  {n:45s} -> {row['pdf_url']}")


if __name__ == "__main__":
    main()
