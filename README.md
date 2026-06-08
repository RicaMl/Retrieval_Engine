# Information Retrieval System for Multi‑Domain Documents

This repository contains the complete implementation of a two‑phase information retrieval system developed for a Kaggle competition). The goal is to rank relevant documents from a large corpus (216 k documents) in response to real‑world queries across five domains: `android`, `programmers`, `unix`, `tex`, `gaming`.

## Overview

The project is split into two complementary notebooks:

| Notebook | Focus | Main techniques |
|----------|-------|----------------|
| `retrieval-engine-competition-phase1.ipynb` | **Phase 1 – Baseline retrieval** | TF‑IDF, BM25+, dense embeddings (all‑MiniLM‑L6‑v2), UMAP visualisation, FAISS indexing. |
| `phase2-3.ipynb` | **Phase 2 – Advanced retrieval with classification** | BGE‑base‑en‑v1.5 embeddings, domain classifier (LinearSVC + SMOTE), category‑restricted search, hybrid BM25+ + dense fusion (RRF), final submission. |

The system progresses from simple lexical matching to semantic dense retrieval and finally to a **classifier‑guided expert mode** that dramatically improves precision and recall while keeping latency low.

## Pipeline

1. **Data loading & preprocessing**  
   - JSON files: `docs.json` (216 041 documents), `queries_train.json` (327 train queries), `queries_test.json` (141 test queries), `qgts_train.json` (ground truth).  
   - Text cleaning: contraction expansion, lowercasing, punctuation removal, stopword removal, **Porter stemming** (used for BM25+ and TF‑IDF).  
   - Stratified 80/20 split of training queries for classifier training/validation.

2. **Lexical retrieval (Phase 1)**  
   - **TF‑IDF** (`TfidfVectorizer`, L2 normalisation) – fast but limited to exact term matches.  
   - **BM25+** (`rank_bm25`) – improved term frequency saturation and document length normalisation.  
   - Both support **category filtering** (when a category is provided).

3. **Dense retrieval (Phase 1 & Phase 2)**  
   - Embedding model: `BAAI/bge-base-en-v1.5` (768 dimensions) – state‑of‑the‑art for semantic search.  
   - Documents encoded only from the `"text"` field (title and tags omitted due to time constraints).  
   - FAISS indexes: a global `IndexFlatIP` for whole‑corpus search, plus **per‑category sub‑indexes** for fast expert‑mode retrieval.  
   - Query encoding uses the BGE instruction prefix:  
     `"Represent this sentence for searching relevant passages: {query}"`.

4. **Domain classifier (Phase 2)**  
   - Trained **only on training queries** (261 after split) – not on documents.  
   - Features: 768‑dim BGE embeddings of the query (with prefix).  
   - Models compared: `LogisticRegression` vs `LinearSVC` (5‑fold cross‑validation). `LinearSVC` wins (accuracy 0.971).  
   - **SMOTE** oversamples minority classes (originally 33‑104 samples) to balance all classes to 104 samples each.  
   - Calibrated probabilities (`CalibratedClassifierCV`) provide confidence scores.

5. **Expert mode vs global mode**  
   - If classifier confidence ≥ threshold (optimised to 0.80) → search only within predicted category (using FAISS sub‑index or temporary BM25 sub‑index).  
   - Else → global search.  
   - This reduces the search space, improves MRR, and avoids fatal errors (confidence threshold benchmark performed on validation set).

6. **Hybrid retrieval (BM25+ + dense + RRF)**  
   - For each query, retrieve top‑`fetch_k` (200) candidates from BM25+ and dense, then fuse with **Reciprocal Rank Fusion** (RRF, k=60).  
   - Prioritises documents that appear in both lists.  
   - Evaluated but ultimately slower and less accurate than pure dense with classifier.

7. **Evaluation & submission**  
   - Metrics: Precision@k, Recall@k, MRR, latency, classifier accuracy.  
   - Final submission uses **embeddings + classifier** with k=500, threshold=0.80 → CSV file ready for Kaggle.

## Results

| Method | P@20 | R@20 | MRR | Latency (ms) | Classifier accuracy |
|--------|------|------|-----|--------------|---------------------|
| TF‑IDF + classifier | 0.63% | 42.60% | 18.61% | 51.8 | 90.91% |
| BM25+ + classifier | 0.76% | 42.79% | 24.80% | 1715 | 90.91% |
| **Embeddings (BGE) + classifier** | **1.24%** | **70.76%** | **52.02%** | **30.6** | **90.91%** |
| Hybrid (BM25 + BGE + RRF) + classifier | 1.11% | 65.85% | 42.72% | 1666 | 90.91% |

- **Best method**: Embeddings (BGE) with classifier → highest recall, MRR, and lowest latency.  
- Classifier accuracy on validation set: **90.91%** (60/66 correct).  
- Hybrid method is **not practical** due to high latency and lower MRR.

## Setup & Dependencies

### Requirements
- Python 3.8+
- Libraries: `sentence-transformers`, `faiss-cpu`, `rank_bm25`, `scikit-learn`, `nltk`, `pandas`, `numpy`, `tqdm`, `contractions`, `matplotlib`, `seaborn`, `umap-learn`.

Install all dependencies with:
```bash
pip install sentence-transformers faiss-cpu rank-bm25 scikit-learn nltk pandas numpy tqdm contractions matplotlib seaborn umap-learn
```

### Data
Place the following JSON files in the paths defined in `Config`:
- `docs.json`
- `queries_train.json`
- `queries_test.json`
- `qgts_train.json`

The notebooks are configured to read from Kaggle input directories; you can adapt the paths in the `Config` class.

## How to Run

1. **Phase 1 notebook**  
   - Run all cells to load data, preprocess, build TF‑IDF/BM25+/dense retrievers, evaluate on validation split, and visualise embeddings with UMAP.

2. **Phase 2 notebook**  
   - Execute cells sequentially: install libraries → load data → define preprocessing → build category index → initialise retrievers → train classifier → benchmark thresholds → evaluate all methods → generate final submission CSV.

The final submission file will be saved as `submission_embeddings_classifier_t0.8_k500.csv` in the output directory.

## Key Design Choices & Limitations

- **Document embeddings use only the `text` field** – title and tags are ignored because encoding the full corpus (216k docs) already takes ~1 h on a GPU. Including title/tags would improve scores but was omitted due to time constraints.  
- **Classifier trained only on queries** – works well for query classification (90.9% accuracy) but would not generalise to documents (not needed).  
- **Fixed confidence threshold** – chosen via validation benchmark; could be made adaptive per category.  
- **No cross‑encoder reranking** – could boost MRR by 5–10 points at the cost of higher latency.

## Future Improvements

- Enrich document content with `title` and `tags` for dense embeddings.  
- Test larger embedding models (`bge-large-en-v1.5`, `e5-mistral-7b`).  
- Implement cross‑encoder reranking on top‑100 candidates.  
- Pre‑compute per‑category BM25 sub‑indexes to make hybrid retrieval faster.

## License

This project is for educational and competition purposes. The datasets are provided by the competition organisers and are not included in this repository.
