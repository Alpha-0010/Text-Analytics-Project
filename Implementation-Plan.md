# Text Analytics Project — Implementation Plan & Approach Document

## Table of Contents

1. [Research Questions](#1-research-questions)
2. [Assigned Task — Implementation Approach (Detailed)](#2-assigned-task)
3. [Corpus Overview](#3-corpus-overview)
4. [Implementation Approaches](#4-implementation-approaches)
   - [Approach A: Classical TF-IDF + Cosine Similarity Pipeline](#approach-a)
   - [Approach B: Semantic Embeddings (Transformer-based)](#approach-b)
   - [Approach C: Topic Modeling (LDA / BERTopic)](#approach-c)
   - [Approach D: Keyword & Lexical Feature Analysis](#approach-d)
   - [Approach E: Hybrid Pipeline (Recommended)](#approach-e)
5. [Comparison of Approaches](#5-comparison-of-approaches)
6. [Recommended Stack & Tools](#6-recommended-stack--tools)

---

## 1. Research Questions

### Primary Research Question

> To what extent do text-based similarities in university strategic plans correlate with external university rankings and institutional classifications?

### Sub-Questions

| #   | Sub-Question                                                                                   |
| --- | ---------------------------------------------------------------------------------------------- |
| 1   | Does text similarity correlate with ranking proximity? _(Primary)_                             |
| 2   | Do same-tier universities show higher textual similarity than cross-tier universities?         |
| 3   | Which linguistic features best predict ranking tier?                                           |
| 4   | Do university types (research-focused vs. teaching-focused) form distinct linguistic clusters? |
| 5   | Do lower-ranked universities mimic the language of higher-ranked universities?                 |

---

## 2. Assigned Task — Implementation Approach (Detailed)

**Assigned to:** Ritik & Shashwat
**Section in Paper:** "Implementation Approach"

### What This Section Must Cover

The Implementation Approach section of the mid-term paper is the technical backbone of the research. It must answer the question: _"How exactly did we (or will we) go from raw PDF strategic plans to meaningful, quantifiable results?"_

The section must cover all of the following in detail:

#### 2.1 Data Acquisition & Preprocessing

- Explain how PDFs are collected (from the corpus spreadsheet — ~40 universities across 5 regions).
- Describe how raw text is extracted from PDFs (tool choices, handling multi-column layouts, tables, headers/footers).
- Describe text cleaning steps: lowercasing, removing boilerplate (page numbers, headers), punctuation removal, stopword removal, lemmatization vs. stemming decision.
- Explain how the QS World Rankings 2026 numerical values are attached to each document (bridging text data with external metadata).
- Address missing/problematic data: documents that are not in English, documents that are scanned images (OCR handling), non-public strategic plan pages.

#### 2.2 Text Representation / Vectorization

- Explain the choice of how to represent each strategic plan as a numerical vector.
- Justify the chosen method(s): TF-IDF, embeddings, or topic distributions.
- Describe the dimensionality of the resulting representation.

#### 2.3 Similarity Computation

- Explain how similarity between any two documents is measured.
- Describe how a pairwise similarity matrix is constructed (N × N matrix for N universities).
- Justify the chosen metric (e.g., cosine similarity, Euclidean distance).

#### 2.4 Clustering

- Explain how universities are grouped based on textual similarity.
- Describe the clustering algorithm(s) used and how the number of clusters is determined.
- Explain how clusters are visualized (dendrograms, heatmaps, 2D/3D scatter).

#### 2.5 Correlation with Rankings

- Explain the statistical method used to correlate text similarity with ranking proximity.
- Define "ranking proximity" — is it absolute rank difference, tier grouping, or something else?
- Explain what a positive vs. negative result means in the context of the research question.

#### 2.6 Linguistic Feature Extraction

- Explain how keywords are extracted per cluster/tier.
- Describe how dominant vocabulary differences between high-ranked and low-ranked universities are identified.

#### 2.7 Evaluation & Validation

- Explain how the results are validated — are clusters meaningful? Is the correlation statistically significant?
- Describe any baselines or sanity checks.

---

## 3. Corpus Overview

| Region            | Assigned To       | Target Count | Ranking Distribution     |
| ----------------- | ----------------- | ------------ | ------------------------ |
| Europe            | Sai Tharun        | 10           | 5 Good, 3 Average, 2 Low |
| North America     | Diksha            | 5            | Mixed                    |
| Asia              | Shravani + Aniket | 10           | Mixed                    |
| Australia/NZ      | Shashwat          | 10           | 5 Good, 3 Average, 2 Low |
| Americas + Africa | Ritik             | 5            | Mixed                    |
| **Total**         |                   | **~40**      |                          |

**Ranking Metric:** QS World University Rankings 2026
**Tier Definition (suggested):**

- Tier 1 (Top): Rank 1–100
- Tier 2 (Mid): Rank 101–500
- Tier 3 (Lower): Rank 501+

---

## 4. Implementation Approaches

---

### Approach A: Classical TF-IDF + Cosine Similarity Pipeline {#approach-a}

**Summary:** The most interpretable, well-understood baseline. Convert each document to a TF-IDF vector, compute pairwise cosine similarity, then statistically correlate those similarity scores with ranking differences.

#### Step-by-Step Pipeline

**Step 1 — Data Collection**

- Download all PDFs listed in the corpus spreadsheet.
- Store locally with a naming convention: `{CountryCode}_{UniversityShortName}_{QSRank}.pdf`

**Step 2 — Text Extraction**

- Use `pdfplumber` or `PyMuPDF (fitz)` to extract raw text from each PDF.
- Handle multi-column layouts by extracting text in reading order.
- Strip headers, footers, page numbers using regex patterns.
- Flag scanned PDFs (where text extraction yields nothing) for OCR via `pytesseract`.

**Step 3 — Text Preprocessing**

```
Raw Text
  → Lowercase
  → Remove non-alphabetic characters (keep hyphens for compound words)
  → Tokenize (split into words)
  → Remove stopwords (English NLTK stopwords + custom domain stopwords like "university", "plan", "strategic")
  → Lemmatize (using spaCy or NLTK WordNetLemmatizer)
  → Rejoin tokens into cleaned document string
```

**Step 4 — TF-IDF Vectorization**

- Use `sklearn.feature_extraction.text.TfidfVectorizer`
- Parameters to tune:
  - `min_df=2` (ignore terms appearing in fewer than 2 documents)
  - `max_df=0.85` (ignore terms appearing in more than 85% of documents — removes corpus-wide boilerplate)
  - `ngram_range=(1,2)` (include unigrams and bigrams like "research excellence")
  - `max_features=5000` (limit vocabulary size)
- Output: Document-term matrix of shape (N_docs × N_features)

**Step 5 — Cosine Similarity Matrix**

- Use `sklearn.metrics.pairwise.cosine_similarity`
- Output: N × N matrix where entry [i,j] = similarity between university i and university j
- Values range from 0 (completely dissimilar) to 1 (identical)

**Step 6 — Ranking Proximity Matrix**

- For each pair (i, j), compute: `|rank_i - rank_j|` (absolute rank difference)
- Invert it to get a "ranking proximity" score: smaller rank difference = more similar in ranking

**Step 7 — Correlation Analysis**

- Flatten both the similarity matrix and the ranking proximity matrix into 1D vectors (upper triangle only, to avoid double-counting)
- Compute **Spearman Rank Correlation** (preferred over Pearson because rankings are ordinal):
  - `scipy.stats.spearmanr(similarity_vector, proximity_vector)`
- Report: correlation coefficient (ρ), p-value, and confidence interval
- If ρ is significantly positive → top-ranked universities write similarly

**Step 8 — Clustering**

- Apply **Hierarchical Agglomerative Clustering** on the TF-IDF vectors using Ward linkage
- Visualize with a **dendrogram** — label each leaf with the university name and QS rank
- Also try **K-Means** (k=3 for the three tiers) and report silhouette score
- Overlay QS tier colors on the dendrogram to visually assess cluster-tier alignment

**Step 9 — Visualization**

- **Heatmap** of the N × N similarity matrix (seaborn, sorted by QS rank)
- **2D Scatter** using PCA or UMAP to reduce TF-IDF vectors to 2D, colored by tier
- **Dendrogram** from hierarchical clustering

**Step 10 — Keyword Analysis per Cluster**

- For each cluster, compute the top-20 TF-IDF terms (average TF-IDF weight across cluster members)
- Compare keyword sets between clusters to identify linguistic markers of each tier

**Strengths:**

- Highly interpretable — keywords are directly readable
- Fast to compute even on 40–50 documents
- Established baseline in text analytics research
- Easy to explain in the paper

**Weaknesses:**

- Bag-of-words assumption — ignores word order and semantics
- "Research" and "Research excellence" are treated as partially different
- Does not capture paraphrasing or semantic similarity (two documents can be highly similar conceptually but score low)
- Sensitive to document length differences

---

### Approach B: Semantic Embeddings (Transformer-based) {#approach-b}

**Summary:** Use pre-trained language models to encode each strategic plan as a dense semantic vector. These embeddings capture meaning, not just word frequency. Two documents that use different words to express the same idea will score high similarity.

#### Step-by-Step Pipeline

**Step 1–3:** Same as Approach A (PDF extraction and preprocessing). Preprocessing is lighter here — keep more text since the model handles semantics.

**Step 4 — Sentence/Document Embedding**

- Use `sentence-transformers` library with a pre-trained model:
  - **`all-mpnet-base-v2`** — highest quality general-purpose embeddings (recommended)
  - **`all-MiniLM-L6-v2`** — faster, slightly lower quality, good for prototyping
- Strategy for long documents (strategic plans can be 20–60 pages):
  - Option 1: **Chunk and Pool** — split document into chunks of 512 tokens, embed each chunk, average all chunk embeddings → 1 vector per document
  - Option 2: **Extractive Summary then Embed** — use `sumy` or `NLTK` to extract the most representative sentences (e.g., top 50), embed those
  - Option 3: **Section-wise Embedding** — embed each identified section (Vision, Mission, Goals) separately, then concatenate or average
- Output: Matrix of shape (N_docs × 768) for mpnet-base

**Step 5 — Cosine Similarity Matrix**

- Same as Approach A: pairwise cosine similarity on the embedding matrix
- `sklearn.metrics.pairwise.cosine_similarity(embeddings)`

**Step 6–10:** Same statistical and visualization steps as Approach A (Spearman correlation, clustering, heatmap, UMAP scatter, keyword extraction).

**For keyword extraction in embedding space:**

- Cannot directly read TF-IDF keywords
- Instead: use `KeyBERT` to extract key phrases per document/cluster by finding n-grams most similar to the document embedding
- Or: fall back to TF-IDF keywords as a complement

**Strengths:**

- Captures semantic similarity — paraphrases and synonyms are handled naturally
- More robust to vocabulary differences between universities in different regions/cultures
- State-of-the-art performance for document similarity tasks
- Better handles idiomatic language in strategic planning documents

**Weaknesses:**

- Harder to interpret — embeddings are 768-dimensional dense vectors
- Computationally heavier (but manageable for ~40 documents)
- Requires a GPU for fast processing (though CPU is feasible at this scale)
- Chunking strategy for long documents is a non-trivial design decision
- May require justification in the paper for why this model was chosen

---

### Approach C: Topic Modeling (LDA / BERTopic) {#approach-c}

**Summary:** Instead of direct similarity, first discover the hidden "topics" discussed across all strategic plans, then represent each university as a distribution over these topics. Similarity is measured between topic distributions rather than raw text vectors.

#### Step-by-Step Pipeline

**Step 1–3:** PDF extraction and preprocessing (same as A, but more aggressive stopword removal is important for LDA quality).

**Step 4 — Topic Modeling**

**Option C1 — LDA (Latent Dirichlet Allocation):**

- Use `gensim.models.LdaModel` or `sklearn.decomposition.LatentDirichletAllocation`
- Preprocess: create Bag-of-Words representation (word counts, not TF-IDF)
- Choose number of topics K (experiment with K = 5, 10, 15, 20)
- Select K using coherence score (`gensim` CoherenceModel) or perplexity
- Output: Each document is a probability distribution over K topics, e.g., [0.4 research, 0.3 innovation, 0.2 community, 0.1 sustainability]

**Option C2 — BERTopic:**

- Uses BERT embeddings + UMAP dimensionality reduction + HDBSCAN clustering
- More modern, produces cleaner topics with actual readable keyphrases
- `from bertopic import BERTopic; model = BERTopic(); topics, probs = model.fit_transform(docs)`
- Does not require pre-specifying K; discovers the number of topics automatically

**Step 5 — Similarity via Topic Distributions**

- For LDA: compute **Jensen-Shannon Divergence** (JSD) between topic distributions of each pair of documents (lower JSD = more similar topic profile)
- Or convert topic vectors to similarity: `1 - JSD(dist_i, dist_j)`
- For BERTopic: use the topic probability vectors as features and compute cosine similarity

**Step 6–9:** Same correlation, clustering, and visualization steps.

**Step 10 — Interpretability**

- Topic models are highly interpretable: each topic has a list of top words
- Can name topics manually: e.g., Topic 1 = "Research & Innovation", Topic 2 = "Community & Inclusion", Topic 3 = "Digital Transformation"
- Analyze which topics are dominant in Tier 1 vs. Tier 3 universities

**Strengths:**

- Highly interpretable topics are excellent for the paper's discussion section
- Reduces dimensionality naturally (K topics instead of thousands of TF-IDF features)
- Reveals thematic structure of the corpus — what strategic plans are "about"
- Useful for answering RQ4 (do university types form distinct linguistic clusters?)

**Weaknesses:**

- LDA is sensitive to preprocessing quality — bad stopword removal → garbage topics
- Choosing the right K requires experimentation and judgment
- LDA has a bag-of-words assumption (same as TF-IDF)
- BERTopic may be overkill and harder to explain for a mid-term paper
- Topic distributions are one step removed from raw text — harder to connect directly to specific phrases in the documents

---

### Approach D: Keyword & Lexical Feature Analysis {#approach-d}

**Summary:** Instead of holistic document similarity, focus on specific, predefined linguistic features. Extract frequencies of key vocabulary categories (e.g., "innovation language", "research language", "inclusion language") and compare those feature vectors across universities and tiers.

#### Step-by-Step Pipeline

**Step 1–3:** PDF extraction and preprocessing.

**Step 4 — Feature Category Definition**
Define a vocabulary for each conceptual category relevant to strategic plans (informed by the literature):

| Category                | Example Keywords                                                     |
| ----------------------- | -------------------------------------------------------------------- |
| Research Excellence     | research, publication, discovery, scholarship, grant, faculty        |
| Innovation              | innovation, entrepreneurship, startup, industry, technology, digital |
| Teaching & Learning     | student, learning, curriculum, pedagogy, graduate, undergraduate     |
| Sustainability          | sustainability, environment, climate, carbon, green                  |
| Inclusion & Diversity   | diversity, equity, inclusion, community, belonging, accessibility    |
| Global Engagement       | international, global, partnership, collaboration, world-class       |
| Leadership & Governance | governance, accountability, leadership, strategy, performance        |

**Step 5 — Feature Vector Construction**

- For each document, count (or TF-IDF weight) all terms in each category
- Normalize by document length
- Output: Each university is represented by a vector of 7 (or more) category scores

**Step 6 — Similarity & Correlation**

- Compute cosine similarity on these compact feature vectors
- Correlate with rankings — which category scores are highest among top-ranked universities?
- Use regression (Lasso/Ridge) to predict ranking tier from category scores

**Step 7 — Statistical Comparison Across Tiers**

- For each category, run a Mann-Whitney U test or ANOVA to see if there are significant differences between Tier 1, Tier 2, and Tier 3 universities
- Visualize with box plots: "Innovation score by QS ranking tier"

**Strengths:**

- Extremely interpretable — directly answers "which language predicts ranking tier?"
- Clean, hypothesis-driven approach that maps well to the research questions
- Results are easy to visualize and present
- Directly addresses RQ3 (which linguistic features best predict ranking tier?) and RQ5 (do lower-ranked universities mimic higher-ranked language?)

**Weaknesses:**

- Requires manually defining vocabulary categories — introduces researcher bias
- Misses vocabulary not in the predefined categories
- Simpler than Approaches A/B — may not satisfy "text analytics methodology" requirements
- Best used as a complement to A or B, not as a standalone method

---

### Approach E: Hybrid Pipeline (Recommended) {#approach-e}

**Summary:** Combine the strengths of Approaches A, C, and D into a single coherent pipeline. Use TF-IDF for the primary similarity analysis (interpretable, explainable), enrich it with topic modeling for thematic insight, and validate findings with targeted keyword analysis.

#### Full Pipeline

```
PDF Collection (40 universities, QS 2026 rankings metadata)
          ↓
Text Extraction (pdfplumber / PyMuPDF)
          ↓
Preprocessing (clean, tokenize, lemmatize, remove stopwords)
          ↓
   ┌──────────────────────────────────────┐
   │   TRACK 1: TF-IDF Similarity         │
   │   • TF-IDF vectorization             │
   │   • Cosine similarity matrix         │
   │   • Hierarchical clustering          │
   │   • Spearman ρ with rank proximity   │
   └──────────────┬───────────────────────┘
                  │
   ┌──────────────────────────────────────┐
   │   TRACK 2: Topic Modeling (LDA)      │
   │   • Discover K topics                │
   │   • Topic distribution per university│
   │   • Cluster by dominant topic        │
   │   • Compare topic mix by tier        │
   └──────────────┬───────────────────────┘
                  │
   ┌──────────────────────────────────────┐
   │   TRACK 3: Keyword Feature Analysis  │
   │   • Score each doc on 7 categories   │
   │   • Tier-wise statistical comparison │
   │   • Regression: category → tier      │
   └──────────────┬───────────────────────┘
                  │
          ┌───────▼────────┐
          │   Triangulate  │
          │   & Discuss    │
          └────────────────┘
```

#### Justification for Hybrid Approach

- **TF-IDF (Track 1)** answers RQ1 and RQ2 directly with a robust, reproducible similarity matrix.
- **Topic Modeling (Track 2)** answers RQ4 by revealing thematic clusters that can be matched to institutional types.
- **Keyword Analysis (Track 3)** answers RQ3 and RQ5 with clear, statistically testable hypotheses.
- The combination allows triangulation — if all three tracks agree, the findings are robust.

#### Recommended Technology Stack

| Task                     | Library/Tool                                     |
| ------------------------ | ------------------------------------------------ |
| PDF Extraction           | `pdfplumber`, `PyMuPDF (fitz)`                   |
| OCR (scanned PDFs)       | `pytesseract` + `pdf2image`                      |
| Preprocessing            | `spaCy` (lemmatization), `NLTK` (stopwords)      |
| TF-IDF                   | `scikit-learn`                                   |
| Cosine Similarity        | `scikit-learn`                                   |
| Clustering               | `scipy` (hierarchical), `scikit-learn` (K-Means) |
| Topic Modeling           | `gensim` (LDA), optional: `bertopic`             |
| Correlation              | `scipy.stats` (Spearman)                         |
| Visualization            | `matplotlib`, `seaborn`, `plotly`                |
| Dimensionality Reduction | `scikit-learn` (PCA), optional: `umap-learn`     |
| Environment              | Python 3.10+, Jupyter Notebook                   |

---

## 5. Comparison of Approaches

| Criterion                            | A: TF-IDF | B: Embeddings | C: Topic Model | D: Keywords | E: Hybrid   |
| ------------------------------------ | --------- | ------------- | -------------- | ----------- | ----------- |
| Interpretability                     | High      | Low           | High           | Very High   | High        |
| Semantic Depth                       | Low       | Very High     | Medium         | Low         | Medium-High |
| Compute Cost                         | Very Low  | Medium        | Low            | Very Low    | Low-Medium  |
| Implementation Complexity            | Low       | High          | Medium         | Low         | Medium      |
| Addresses RQ1 (similarity ↔ ranking) | ✓         | ✓             | Partial        | ✗           | ✓           |
| Addresses RQ3 (which features?)      | Partial   | ✗             | ✓              | ✓           | ✓           |
| Addresses RQ4 (institutional types)  | ✓         | ✓             | ✓              | Partial     | ✓           |
| Suitable for mid-term paper          | ✓         | Partial       | ✓              | ✓           | ✓           |
| Best for final paper?                | Baseline  | Advanced      | Complement     | Complement  | Yes         |

---

## 6. Recommended Stack & Tools

```
pip install pdfplumber pymupdf spacy nltk scikit-learn scipy gensim
pip install matplotlib seaborn plotly umap-learn
python -m spacy download en_core_web_sm
```

### Suggested Project Structure

```
text-analytics-project/
│
├── data/
│   ├── raw_pdfs/          # Downloaded strategic plan PDFs
│   ├── extracted_text/    # .txt files per university
│   └── metadata.csv       # University name, region, QS rank, tier
│
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_tfidf_similarity.ipynb
│   ├── 04_topic_modeling.ipynb
│   └── 05_results_visualization.ipynb
│
├── src/
│   ├── extract.py         # PDF → text
│   ├── preprocess.py      # Cleaning pipeline
│   ├── vectorize.py       # TF-IDF + embeddings
│   ├── similarity.py      # Similarity matrix + correlation
│   └── visualize.py       # All plots
│
└── outputs/
    ├── similarity_matrix.csv
    ├── cluster_assignments.csv
    └── figures/
```

---

_This document is intended to serve as the foundation for the Implementation Approach section of the mid-term paper and as a working guide for Ritik and Shashwat._
