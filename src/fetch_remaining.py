"""
fetch_remaining.py
------------------
Handles the two categories of missing documents:

  1. PDF URLs returning 403  → retry with full browser-simulation headers
  2. HTML page URLs          → scrape page text directly with BeautifulSoup
                               and save straight to extracted_text/ (no PDF needed)

Usage:
    python src/fetch_remaining.py

Requires: requests, beautifulsoup4  (both already installed)
"""

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
PDF_DIR  = BASE / "data" / "raw_pdfs"
TEXT_DIR = BASE / "data" / "extracted_text"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

# ── full browser-simulation headers ──────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

# Tags whose content we want to keep when scraping HTML
KEEP_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "li", "td", "th", "blockquote"]

# Tags to remove entirely before parsing
STRIP_TAGS = ["nav", "header", "footer", "script", "style", "noscript",
              "iframe", "aside", "form", "button", "svg", "img", "figure",
              "figcaption", "video", "audio"]


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


# ── PDF download with full headers ────────────────────────────────────────────
def try_download_pdf(url: str, dest: Path) -> bool:
    session = requests.Session()
    # First hit the base domain to pick up any cookies
    base = "/".join(url.split("/")[:3])
    try:
        session.get(base, headers=HEADERS, timeout=15)
    except Exception:
        pass

    try:
        r = session.get(url, headers={**HEADERS, "Referer": base},
                        timeout=30, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except requests.HTTPError as e:
        print(f"    HTTP {e.response.status_code}")
    except Exception as e:
        print(f"    {e}")
    return False


# ── HTML scraping ─────────────────────────────────────────────────────────────
def scrape_html_to_text(url: str) -> str:
    """
    Fetch an HTML page and extract clean body text.
    Returns empty string if the page is unreachable or yields < 200 words.
    """
    session = requests.Session()
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"    Fetch failed: {e}")
        return ""

    soup = BeautifulSoup(r.text, "html.parser")

    # Remove boilerplate structural elements
    for tag in soup(STRIP_TAGS):
        tag.decompose()

    # Extract text from content-bearing tags
    chunks = []
    for tag in soup.find_all(KEEP_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if len(text.split()) >= 4:        # skip very short fragments
            chunks.append(text)

    # Fallback: grab all remaining body text
    if len(" ".join(chunks).split()) < 200:
        body = soup.find("body")
        if body:
            chunks = [body.get_text(separator="\n", strip=True)]

    full_text = "\n".join(chunks)
    # Collapse excessive whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)
    return full_text.strip()


# ── extract text from a newly downloaded PDF ─────────────────────────────────
def extract_pdf_to_text(pdf_path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=3, y_tolerance=3)
            if t:
                pages.append(t)
    return "\n".join(pages).strip()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(BASE / "data" / "metadata.csv")

    # Find universities still missing a .txt file
    missing = []
    for _, row in df.iterrows():
        txt = TEXT_DIR / (safe_filename(row["university"]) + ".txt")
        if not txt.exists():
            missing.append(row)

    if not missing:
        print("All documents already present - nothing to do.")
        return

    print(f"Fetching {len(missing)} missing documents...\n")
    succeeded, failed = [], []

    for row in missing:
        name = row["university"]
        url  = row["pdf_url"]
        safe = safe_filename(name)
        txt_path = TEXT_DIR / (safe + ".txt")
        is_pdf_url = url.lower().endswith(".pdf")

        print(f"[{'PDF' if is_pdf_url else 'HTML'}] {name}")

        # ── Branch 1: direct PDF URL ──────────────────────────────────────────
        if is_pdf_url:
            pdf_dest = PDF_DIR / (safe + ".pdf")
            print(f"  Downloading PDF...")
            ok = try_download_pdf(url, pdf_dest)
            if ok and pdf_dest.stat().st_size > 1024:
                print(f"  Extracting text from PDF...")
                try:
                    text = extract_pdf_to_text(pdf_dest)
                    words = len(text.split())
                    if words > 100:
                        txt_path.write_text(text, encoding="utf-8")
                        print(f"  OK - {words:,} words")
                        succeeded.append(name)
                    else:
                        print(f"  WARN - only {words} words extracted (scanned PDF?)")
                        failed.append(name)
                except Exception as e:
                    print(f"  PDF extract error: {e}")
                    failed.append(name)
            else:
                pdf_dest.unlink(missing_ok=True)
                print(f"  FAIL - could not download PDF")
                failed.append(name)

        # ── Branch 2: HTML page ───────────────────────────────────────────────
        else:
            print(f"  Scraping HTML...")
            text = scrape_html_to_text(url)
            words = len(text.split())
            if words >= 200:
                txt_path.write_text(text, encoding="utf-8")
                print(f"  OK - {words:,} words")
                succeeded.append(name)
            else:
                print(f"  FAIL - only {words} words (JS-rendered page or access denied)")
                failed.append(name)

        time.sleep(1.5)   # polite delay

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Succeeded : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print(f"\n  Still needs manual download:")
        for n in failed:
            row = next(r for r in missing if r["university"] == n)
            print(f"    {n}")
            print(f"      {row['pdf_url']}")
    total_txt = len(list(TEXT_DIR.glob("*.txt")))
    print(f"\n  Total documents ready: {total_txt}/43")


if __name__ == "__main__":
    main()
