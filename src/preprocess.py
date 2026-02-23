"""
preprocess.py
-------------
Text cleaning and normalisation pipeline for strategic plan documents.

Provides:
  - preprocess_text(text)  → cleaned, lemmatized token string
  - preprocess_corpus(df)  → DataFrame with a 'processed_text' column added

Usage (standalone):
    python src/preprocess.py

Requires: nltk, spacy (en_core_web_sm)
"""

import re
import string
from pathlib import Path

import nltk
import spacy

# ── NLTK data (download once) ────────────────────────────────────────────────
for resource in ["stopwords", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt"
                       else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords

# ── spaCy model ───────────────────────────────────────────────────────────────
try:
    NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Run:  python -m spacy download en_core_web_sm"
    )

# ── Stop-word set (NLTK base + domain-specific) ───────────────────────────────
DOMAIN_STOPWORDS = {
    # generic document noise
    "university", "universities", "college", "colleges", "institution",
    "strategic", "strategy", "plan", "planning", "plans",
    "goal", "goals", "objective", "objectives", "vision", "mission",
    "value", "values", "priority", "priorities",
    # numbers-as-words that appear frequently
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third",
    # boilerplate filler
    "will", "shall", "also", "may", "ensure", "support", "provide",
    "develop", "include", "continue", "work", "make", "use",
    "year", "years", "page", "section", "table", "figure",
}

STOP_WORDS = set(stopwords.words("english")) | DOMAIN_STOPWORDS


# ── core function ─────────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a single document string.

    Pipeline:
        1. Lowercase
        2. Remove URLs, emails, numbers-only tokens
        3. Remove punctuation (keep intra-word hyphens)
        4. Tokenise and lemmatize via spaCy
        5. Remove stopwords and very short tokens (< 3 chars)

    Returns a single space-separated string of lemmatized tokens.
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs and emails
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)

    # 3. Remove standalone numbers (page numbers, years, percentages)
    text = re.sub(r"\b\d+\.?\d*%?\b", " ", text)

    # 4. Remove punctuation except intra-word hyphens
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)  # remove leading/trailing hyphens

    # 5. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Lemmatize via spaCy (processes up to 1 million chars by default)
    doc = NLP(text[:1_000_000])  # guard against extremely long documents
    tokens = [
        token.lemma_
        for token in doc
        if (
            token.is_alpha
            and len(token.lemma_) >= 3
            and token.lemma_ not in STOP_WORDS
            and not token.is_stop
        )
    ]

    return " ".join(tokens)


# ── corpus-level helper ───────────────────────────────────────────────────────

def safe_filename(name: str) -> str:
    """Convert a university name to a safe filesystem filename."""
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


def preprocess_corpus(df, text_dir: Path):
    """
    Given a metadata DataFrame and the path to extracted .txt files,
    return the DataFrame with a new 'processed_text' column.

    Parameters
    ----------
    df        : pd.DataFrame  — must have a 'university' column
    text_dir  : Path          — directory containing <SafeName>.txt files
    """
    import pandas as pd

    processed_texts = []
    for _, row in df.iterrows():
        txt_path = text_dir / (safe_filename(row["university"]) + ".txt")  # noqa: F821
        if txt_path.exists():
            raw = txt_path.read_text(encoding="utf-8")
            processed_texts.append(preprocess_text(raw))
        else:
            print(f"[WARN] No extracted text for: {row['university']}")
            processed_texts.append("")

    df = df.copy()
    df["processed_text"] = processed_texts
    return df


# ── standalone run ────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    BASE = Path(__file__).parent.parent
    metadata_path = BASE / "data" / "metadata.csv"
    text_dir = BASE / "data" / "extracted_text"
    out_path = BASE / "data" / "processed_corpus.csv"

    df = pd.read_csv(metadata_path)
    print(f"Preprocessing {len(df)} documents …\n")

    df = preprocess_corpus(df, text_dir)

    filled = (df["processed_text"] != "").sum()
    print(f"\nProcessed {filled}/{len(df)} documents successfully.")

    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
