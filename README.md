# NLP Tasks — Overview

Two mentor-assigned NLP projects, each self-contained with its own pipeline, data, and outputs.

> For full implementation details, refer to the `README.md` inside each task folder.

---

## Repository Structure

```
NLP/
├── README.md                  # This file — top-level overview
├── Task1/
│   ├── README.md              # Task 1 detailed documentation
│   ├── 01_research_and_data_context.ipynb
│   ├── 02_preprocessing_and_language_detection.ipynb
│   ├── 03_full_processing_pipeline.ipynb
│   ├── 04_model_training_and_comparison.ipynb
│   ├── tag1.csv
│   ├── sample.csv
│   ├── models/
│   └── Output/
│       ├── df_with_idx.csv
│       └── document_embeddings.npy
└── Task2/
    ├── README.md              # Task 2 detailed documentation
    ├── processing.ipynb
    ├── prediction_code.ipynb
    ├── steps_to_perform.ipynb
    ├── Data/
    │   └── harry_potter_books.csv
    └── Outputs/
        ├── word2vec_model.keras
        └── embeddings1.pkl
```

---

## Task 1 — Multilingual Document Processing & Classification

Full ML workflow applied to a multilingual tagged document dataset.

| Area | Details |
|---|---|
| Input | `Task1/tag1.csv` |
| Focus | Language detection, text cleaning, embedding generation, classification |
| Models | Logistic regression baseline + comparisons |
| Outputs | `Output/df_with_idx.csv`, `Output/document_embeddings.npy` |

**Run order:**
```
01_research_and_data_context.ipynb
02_preprocessing_and_language_detection.ipynb
03_full_processing_pipeline.ipynb
04_model_training_and_comparison.ipynb
```

---

## Task 2 — Word2Vec Embeddings on Harry Potter Corpus

Custom Word2Vec model trained from scratch on Harry Potter book text.

| Area | Details |
|---|---|
| Input | `Task2/Data/harry_potter_books.csv` |
| Focus | Corpus prep, vocabulary building, skip-gram + negative sampling, semantic analysis |
| Model | TensorFlow/Keras embedding layer (300 dims) |
| Outputs | `Outputs/word2vec_model.keras`, `Outputs/embeddings1.pkl` |

**Run order:**
```
processing.ipynb
prediction_code.ipynb
steps_to_perform.ipynb   ← optional (planning notes)
```

---

## Setup

Each task has its own dependencies. Install them separately to avoid conflicts.

```bash
# Task 1
pip install -r Task1/requirements.txt

# Task 2
pip install -r Task2/requirements.txt
```

If running both tasks in a single environment, install both requirements files sequentially.

---

## Notes

- Each task is fully self-contained — data, notebooks, and outputs are isolated within their respective folders.
- This file is the entry point. Open the task-level `README.md` for step-by-step pipeline details.