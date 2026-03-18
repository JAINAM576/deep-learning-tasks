# Task 1 — Multilingual Document Processing & Classification

End-to-end pipeline for loading, cleaning, language detection, embedding, and classifying a multilingual tagged document dataset.

---

## Table of Contents

- [Data Source](#data-source)
- [Project Structure](#project-structure)
- [Workflow Overview](#workflow-overview)
- [Notebooks](#notebooks)
- [Utility Scripts](#utility-scripts)
- [Output Files](#output-files)
- [Recommended Run Order](#recommended-run-order)
- [Dependencies](#dependencies)

---

## Data Source

| File | Description |
|---|---|
| `tag1.csv` | Main raw/source dataset — starting point for the entire pipeline |
| `sample.csv` | Deduplicated and lightly processed sample used for checks and experimentation |

---

## Project Structure

```
Task1/
├── 01_research_and_data_context.ipynb            # Research context, data intro, initial inspection
├── 02_preprocessing_and_language_detection.ipynb # Preprocessing, deduplication, language checks
├── 03_full_processing_pipeline.ipynb             # Full pipeline: cleaning, embedding, export, validation
├── 04_model_training_and_comparison.ipynb        # Model training and comparison results
├── application_methods.ipynb                     # Application-side notes and method ideas
├── language_detect.ipynb                         # Standalone language detection experiments
├── preprocessing.ipynb                           # Original combined notebook (all steps in one place)
├── understanding.ipynb                           # Additional analysis and understanding notes
├── README.md                                    
├── requirements.txt                             # Project requirements file
├── tag1.csv                                      # Raw source dataset
├── sample.csv                                    # Deduplicated sample dataset
├── models/
│   └── model_without_feature_eng_logistic.ipynb  # Logistic regression baseline (no feature engineering)
└── Output/
    ├── df_with_idx.csv                           # Processed rows with embedding index mapping
    └── document_embeddings.npy                   # Final document embedding vectors
```

---

## Workflow Overview

The four primary notebooks form a linear pipeline:

```
01  Research & Context
        ↓
02  Preprocessing & Language Detection
        ↓
03  Full Processing Pipeline
        ↓
04  Model Training & Comparison
```

---

## Notebooks

### Primary Notebooks

#### `01_research_and_data_context.ipynb`
Entry point for the project. Contains:
- Domain research notes and background context
- Initial data loading from `tag1.csv`
- Early inspection: shape, distributions, null counts, sample rows

#### `02_preprocessing_and_language_detection.ipynb`
Handles the first wave of data cleaning. Covers:
- Duplicate detection and removal
- Basic text normalisation
- Language detection across multiple methods
- Exploratory checks on cleaned data

#### `03_full_processing_pipeline.ipynb`
The core processing notebook. Runs the complete pipeline:
- Text cleaning and encoding fixes
- Language assignment based on detection results
- Document embedding generation
- Processed data export to `Output/`
- Similarity checks and validation

#### `04_model_training_and_comparison.ipynb`
Final modelling stage. Includes:
- Model training using processed embeddings
- Evaluation metrics and performance comparison across models
- Summary of results

---

### Supporting Notebooks

#### `preprocessing.ipynb`
Original monolithic notebook with all stages combined. Kept as a reference and fallback; the four primary notebooks above are the cleaned, modular version of this file.

#### `language_detect.ipynb`
Standalone experiments in language detection. Explores and benchmarks multiple detection methods independently of the main pipeline.

#### `understanding.ipynb`
Supplementary analysis and exploratory notes. Used for deeper inspection of data patterns and intermediate outputs.

#### `application_methods.ipynb`
Notes and experiments around application-side ideas — potential deployment approaches and method evaluations.

---

## Output Files

| File | Description |
|---|---|
| `Output/df_with_idx.csv` | Processed dataset with selected columns and embedding index mapping |
| `Output/document_embeddings.npy` | NumPy array of document embedding vectors |

> **Note:** `04_model_training_and_comparison.ipynb` and any downstream analysis depend on both output files being present. Run `03_full_processing_pipeline.ipynb` first to generate them.

---

## Recommended Run Order

```
1. 01_research_and_data_context.ipynb            # Understand the data
2. 02_preprocessing_and_language_detection.ipynb # Clean and detect languages
3. 03_full_processing_pipeline.ipynb             # Generate embeddings and exports
4. 04_model_training_and_comparison.ipynb        # Train and evaluate models
```

The supporting notebooks (`preprocessing.ipynb`, `language_detect.ipynb`, `understanding.ipynb`, `application_methods.ipynb`) can be consulted independently at any stage.

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, manipulation, and array handling |
| `langdetect` / `langid` | Language detection |
| `sentence-transformers` or equivalent | Document embedding generation |
| `scikit-learn` | Model training, evaluation, and similarity computation |
| `matplotlib` / `seaborn` | Exploratory visualisation |
