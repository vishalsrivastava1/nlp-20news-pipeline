---

## `ARCHITECTURE.md`

```md
# Architecture

## System Overview
This project implements a modular NLP pipeline for document classification and semantic clustering using both sparse and dense representations.

The architecture consists of three main processing pipelines:

1. Classic sparse feature classification
2. Dense embedding classification
3. Unsupervised clustering with LLM topic labeling

---

## Data Flow

Raw Text
↓
Dataset Loader (data.py)
↓
Train/Test Split
↓
| Part 1: TF-IDF Pipeline |
| Text → Vectorizer → Classifier |

↓
Metrics + Confusion Reports

| Part 2: Embedding Pipeline |
| Text → SentenceTransformer → Model |

↓
Metrics + Confusion Reports

| Part 3: Clustering Pipeline |
| Text → Embeddings → KMeans → LLM |

↓
Topic Tree Output


---

## Modules

### `data.py`
- Loads 20 Newsgroups dataset
- Removes headers/footers/quotes
- Selects 10,000 documents
- Performs stratified train/test split

---

### `utils.py`
- Evaluation metrics (accuracy, macro-F1)
- Confusion matrix generation
- Top confusion pair extraction

---

### `part1_classic.py`
Implements classic text classification pipeline:

- TF-IDF / Bag-of-Words vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- scikit-learn Pipeline used to prevent data leakage

---

### `part2_embeddings.py`
Implements embedding-based classification:

- SentenceTransformer embedding generation
- Dense feature training
- Gaussian Naive Bayes baseline
- Logistic Regression, Linear SVM, Random Forest

---

### `part3_topic_tree.py`
Implements semantic clustering:

- Document embedding
- KMeans clustering with K < 10
- Elbow method for cluster selection
- Representative document selection
- Second-level clustering on largest clusters

---

### `llm_labeler.py`
- Constructs prompts for cluster labeling
- Calls OpenAI API
- Generates structured topic labels

---

## Design Decisions

### Pipeline Usage
scikit-learn pipelines ensure no information leakage between training and test data.

### Sparse vs Dense Representations
Sparse features capture lexical patterns, while dense embeddings capture semantic similarity.

### Elbow Method
Used to select the optimal number of clusters based on diminishing inertia improvement.

### LLM Labeling
LLM-generated labels provide human-interpretable topic names for clusters.

---

## Outputs

- Classification metrics
- Confusion reports
- Cluster inertia values
- Two-level topic tree

Outputs are stored in the `outputs/` directory.
