"""
smart_fetch.py
--------------
For universities whose URLs are stale (404) or DNS-unreachable, this script:
  1. Fetches the university's main homepage
  2. Crawls links that look like a strategic plan page
  3. Extracts and saves the text content to extracted_text/

Also retries the 4 Australian 403 PDFs with a session + cookie warm-up.

Usage:
    python src/smart_fetch.py
"""

import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

BASE     = Path(__file__).parent.parent
TEXT_DIR = BASE / "data" / "extracted_text"
PDF_DIR  = BASE / "data" / "raw_pdfs"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

STRIP_TAGS = ["nav", "header", "footer", "script", "style", "noscript",
              "iframe", "aside", "form", "button", "svg"]
KEEP_TAGS  = ["p", "h1", "h2", "h3", "h4", "h5", "li", "td", "blockquote"]

# Keywords that identify a strategic plan page
PLAN_KEYWORDS = [
    "strategic-plan", "strategic_plan", "strategicplan",
    "strategy", "vision", "mission", "our-plan", "future",
    "long-range", "bold-aspirations", "path-forward",
]


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


def get(session, url, timeout=20):
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r
    except Exception:
        return None


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    chunks = []
    for tag in soup.find_all(KEEP_TAGS):
        text = tag.get_text(separator=" ", strip=True)
        if len(text.split()) >= 4:
            chunks.append(text)
    if len(" ".join(chunks).split()) < 200:
        body = soup.find("body")
        if body:
            chunks = [body.get_text(separator="\n", strip=True)]
    full = "\n".join(chunks)
    full = re.sub(r"\n{3,}", "\n\n", full)
    full = re.sub(r"[ \t]{2,}", " ", full)
    return full.strip()


def find_plan_links(html: str, base_url: str) -> list[str]:
    """Find links that look like strategic plan pages."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        text = a.get_text(strip=True).lower()
        combined = href + " " + text
        if any(kw in combined for kw in PLAN_KEYWORDS):
            full = urljoin(base_url, a["href"])
            if urlparse(full).scheme in ("http", "https"):
                candidates.append(full)
    # Deduplicate while preserving order
    seen = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def crawl_for_text(session, main_url: str, depth: int = 2) -> str:
    """
    Fetch main_url, hunt for strategic-plan links, fetch those pages,
    and return the best (longest) text found across all pages visited.
    """
    visited = set()
    best_text = ""

    def fetch_and_extract(url: str, current_depth: int):
        nonlocal best_text
        if url in visited or current_depth < 0:
            return
        visited.add(url)

        r = get(session, url)
        if r is None:
            return

        ct = r.headers.get("content-type", "")
        # If it redirected to a PDF, download it
        if "pdf" in ct or url.lower().endswith(".pdf"):
            dest = PDF_DIR / (safe_filename(url.split("/")[-1]) + ".pdf")
            dest.write_bytes(r.content)
            return

        text = html_to_text(r.text)
        if len(text.split()) > len(best_text.split()):
            best_text = text

        if current_depth > 0:
            for link in find_plan_links(r.text, url)[:8]:  # limit breadth
                fetch_and_extract(link, current_depth - 1)

    fetch_and_extract(main_url, depth)
    return best_text


def try_pdf_403(name: str, url: str) -> bool:
    """Retry a 403 PDF with cookie warm-up."""
    session = requests.Session()
    base = "/".join(url.split("/")[:3])
    session.get(base, headers=HEADERS, timeout=15)   # warm-up cookies
    try:
        r = session.get(url, headers={**HEADERS, "Referer": base},
                        timeout=30, stream=True)
        r.raise_for_status()
        dest = PDF_DIR / (safe_filename(name) + ".pdf")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception:
        return False


# ── University targets with main-domain fallback ──────────────────────────────
# Format: (display_name, main_url_to_crawl)
HTML_TARGETS = [
    ("University of Hong Kong",       "https://www.hku.hk/"),
    ("Hong Kong Polytechnic University", "https://www.polyu.edu.hk/en/about-polyu/strategic-plan/"),
    ("Universiti Malaya",             "https://www.um.edu.my/"),
    ("Stanford University",           "https://www.stanford.edu/"),
    ("Harvard University",            "https://www.harvard.edu/"),
    ("University of Michigan",        "https://umich.edu/"),
    ("University of Kansas",          "https://ku.edu/"),
    ("Texas Tech University",         "https://www.ttu.edu/"),
    ("MIT",                           "https://web.mit.edu/"),
    ("Princeton University",          "https://www.princeton.edu/"),
    ("University of Toronto",         "https://www.utoronto.ca/"),
    ("University of Cape Town",       "https://www.uct.ac.za/"),
    ("University of Witwatersrand",   "https://www.wits.ac.za/"),
    ("Universidad de los Andes",      "https://uniandes.edu.co/"),
    ("Auburn University",             "https://www.auburn.edu/"),
    ("Miami University",              "https://miamioh.edu/"),
]

PDF_403_TARGETS = [
    ("University of Otago",     "https://www.otago.ac.nz/__data/assets/pdf_file/0029/314885/download-pae-tata-strategic-plan-to-2030-0245908.pdf"),
    ("Deakin University",       "https://www.deakin.edu.au/__data/assets/pdf_file/0020/115157/deakin-university-strategic-plan.pdf"),
    ("University of Newcastle", "https://www.newcastle.edu.au/__data/assets/pdf_file/0008/607292/Strategic-Plan-2020-2025.pdf"),
    ("University of Tasmania",  "https://www.utas.edu.au/__data/assets/pdf_file/0007/1794652/University-of-Tasmania-Strategic-Plan-2025-Refresh.pdf"),
]


def main():
    import pdfplumber

    succeeded, failed = [], []

    # ── Phase 1: retry 403 PDFs ───────────────────────────────────────────────
    print("=== Phase 1: Retrying 403 PDFs ===\n")
    for name, url in PDF_403_TARGETS:
        txt_path = TEXT_DIR / (safe_filename(name) + ".txt")
        if txt_path.exists():
            print(f"[SKIP] {name}")
            continue
        print(f"[PDF] {name}")
        ok = try_pdf_403(name, url)
        if ok:
            dest = PDF_DIR / (safe_filename(name) + ".pdf")
            if dest.exists() and dest.stat().st_size > 1024:
                with pdfplumber.open(dest) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages).strip()
                words = len(text.split())
                if words > 100:
                    txt_path.write_text(text, encoding="utf-8")
                    print(f"  OK - {words:,} words")
                    succeeded.append(name)
                    time.sleep(1)
                    continue
        print(f"  FAIL - still blocked")
        failed.append(name)
        time.sleep(1)

    # ── Phase 2: HTML crawl from main domain ──────────────────────────────────
    print("\n=== Phase 2: HTML crawl from main domains ===\n")
    for name, main_url in HTML_TARGETS:
        txt_path = TEXT_DIR / (safe_filename(name) + ".txt")
        if txt_path.exists():
            print(f"[SKIP] {name}")
            continue
        print(f"[HTML] {name}")
        session = requests.Session()
        text = crawl_for_text(session, main_url, depth=2)
        words = len(text.split())
        if words >= 300:
            txt_path.write_text(text, encoding="utf-8")
            print(f"  OK - {words:,} words")
            succeeded.append(name)
        else:
            print(f"  FAIL - only {words} words")
            failed.append(name)
        time.sleep(1.5)

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(list(TEXT_DIR.glob("*.txt")))
    print(f"\n{'='*55}")
    print(f"  Succeeded this run : {len(succeeded)}")
    print(f"  Failed this run    : {len(failed)}")
    print(f"  Total docs ready   : {total}/43")
    if failed:
        print(f"\n  Needs manual download:")
        for n in failed:
            print(f"    - {n}")


if __name__ == "__main__":
    main()
