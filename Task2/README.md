# Task 2 — Harry Potter Word2Vec Embeddings

Train a Word2Vec-style embedding model on Harry Potter book text, then use the learned vectors for semantic similarity analysis and visualization.

---

## Table of Contents

- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Notebooks](#notebooks)
- [Output Files](#output-files)
- [Recommended Run Order](#recommended-run-order)
- [Dependencies](#dependencies)

---

## Data Source

| Field | Details |
|---|---|
| File | `Data/harry_potter_books.csv` |
| Contents | Book-wise text from the Harry Potter series |

---

## Project Structure

```
Task2/
├── processing.ipynb           # End-to-end preprocessing + Word2Vec training
├── prediction_code.ipynb      # Embedding analysis: similarity checks & t-SNE plots
├── steps_to_perform.ipynb     # Planning notes and NLP pipeline checklist
├── README.md                 
├── requirements.txt           # Project requirements file
├── Data/
│   └── harry_potter_books.csv # Source dataset
└── Outputs/
    ├── word2vec_model.keras   # Trained Keras embedding model
    └── embeddings1.pkl        # Saved embeddings + vocabulary mappings
```

---

## Pipeline Overview

The end-to-end workflow in `processing.ipynb` follows these stages:

### 1. Data Loading & Inspection
- Load `harry_potter_books.csv` using `pandas`
- Inspect dataset shape, book count, and max text lengths per book

### 2. Corpus Creation
- Concatenate all book text into a single corpus string
- Measure total corpus length and word count

### 3. Text Cleaning
- Lowercase the full corpus
- Replace newlines with spaces
- Apply regex-based noise removal

### 4. Sentence Tokenization
- Split corpus into sentences using NLTK (`punkt` / `punkt_tab`)
- Inspect sentence count and sample outputs
- *(Commented alternatives: spaCy, Stanza)*

### 5. Punctuation Cleanup
- Strip non-alphabetic characters from each sentence using regex
- Build the `clean_sentences` list

### 6. Word Tokenization & Stopwords
- Tokenize sentences with `word_tokenize`
- Load NLTK stopwords and extend with custom entries: `said`, `could`, `would`, `might`, `also`, `one`
- Compute approximate token statistics

### 7. Vocabulary Building
- Count word frequencies with `Counter`
- Build `word2idx` and `idx2word` mappings
- Add special tokens: `<PAD>` (index `0`) and `<UNK>` (unseen words)

### 8. Sentence Encoding
- Convert tokenized sentences into index sequences for model input

### 9. Skip-gram Pair Generation
- Generate target–context pairs with a dynamic window (`max_window = 5`)
- Apply subsampling via word-frequency-based keep probabilities

### 10. Negative Sampling
- Build positive pairs: `(target, context, label=1)`
- Generate negative pairs by random context substitution
- Combine into a binary classification training set

### 11. Model Training
- Build a TensorFlow/Keras model with a shared `Embedding` layer (300 dimensions)
- Train using target/context inputs and binary labels across multiple epochs
- *(Commented alternative: NCE loss training block)*

### 12. Embedding Extraction & Similarity
- Extract the learned embedding matrix from the trained layer
- Utility functions:
  - `get_embedding(word)` — retrieve the vector for a word
  - `most_similar(word, top_n)` — find nearest neighbours using cosine similarity

### 13. Saving Artifacts
- Save trained model as `Outputs/word2vec_model.keras`
- Serialize embeddings and vocabulary mappings to `Outputs/embeddings1.pkl`

---

## Notebooks

### `processing.ipynb` — Main Training Notebook
Covers the full pipeline from raw CSV to a trained Word2Vec model. Run this first.

### `prediction_code.ipynb` — Analysis Notebook
Post-training analysis using saved artifacts:
- Load embeddings and mappings from `Outputs/embeddings1.pkl`
- Query top-N similar words for any term
- Compute pairwise cosine similarity between words
- Visualize word vectors with t-SNE

### `steps_to_perform.ipynb` — Planning Notes
Conceptual pipeline reference:

> Collection → Cleaning → Tokenization → Vocabulary → Vectorization

Includes notes on optional steps such as stemming/lemmatization, rare/high-frequency word filtering, and sequence padding.

---

## Output Files

| File | Description |
|---|---|
| `Outputs/word2vec_model.keras` | Trained Keras embedding model |
| `Outputs/embeddings1.pkl` | Pickle file with embedding matrix, `word2idx`, and `idx2word` |

> **Note:** `prediction_code.ipynb` requires `Outputs/embeddings1.pkl` to be present before it can run.

---

## Recommended Run Order

```
1. processing.ipynb          # Train the model and generate output artifacts
2. prediction_code.ipynb     # Analyse embeddings and run similarity queries
3. steps_to_perform.ipynb    # Reference / planning notes (optional)
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib` | Plotting and t-SNE visualisation |
| `nltk` | Sentence tokenisation, word tokenisation, stopwords |
| `tensorflow` / `keras` | Embedding model construction and training |
| `scikit-learn` | Cosine similarity computation |
| `pickle` | Serialising trained embeddings and vocabulary |