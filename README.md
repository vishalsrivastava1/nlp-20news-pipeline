# 20 Newsgroups NLP Pipeline (Classification + Topic Clustering)

## Overview
This project builds an end-to-end NLP pipeline on a 10,000-document subset of the **scikit-learn 20 Newsgroups dataset**.

The pipeline includes:

1. **Classic sparse features (BoW/TF-IDF) + supervised classification**
2. **SentenceTransformer embeddings + supervised classification**
3. **Semantic clustering (<10 clusters) + 2-level topic tree with LLM-generated labels**

The project compares lexical vs semantic representations and demonstrates unsupervised topic discovery.

---

## Dataset
- Source: `sklearn.datasets.fetch_20newsgroups`
- Documents: 10,000
- Classes: 20 categories
- Preprocessing: headers, footers, quotes removed

---

## Project Structure
```text
nlp-20news-pipeline/
│
├── src/
│ ├── data.py
│ ├── utils.py
│ ├── part1_classic.py
│ ├── part2_embeddings.py
│ ├── part3_topic_tree.py
│ └── llm_labeler.py
│
├── outputs/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add OpenAI API KEY to .env
```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Run Part 1 Classic features
```bash
python -m src.part1_classic
```

### 5. Run Part 2 SentenseTransformer Embeddings 
```bash
python -m src.part2_embeddings
```

### 6. Run Part 3 Topic Clustering
```bash
python -m src.part3_topic_tree
```

---

## Technologies Used

-Python
-Scikit-learn
-SentenceTransformers
-OpenAI API
-KMeans clustering

---

## Key Findings

-TF-IDF + Linear SVM performs best for topic classification.
-Dense embeddings capture semantic similarity but may reduce performance on vocabulary-driven categories.
-Semantic clustering produces meaningful topic groupings.

