import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from .data import load_20newsgroups_10k, split_stratified
from .utils import eval_metrics, confusion_mat, top_confusions


def embed_texts(model_name, texts, batch_size=64):
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(emb)


def run(output_dir="outputs", embed_model="all-MiniLM-L6-v2", random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    X, y, label_names = load_20newsgroups_10k(random_state=random_state)
    Xtr_txt, Xte_txt, ytr, yte = split_stratified(X, y, random_state=random_state)

    # 1) Convert text -> dense vectors
    Xtr = embed_texts(embed_model, Xtr_txt)
    Xte = embed_texts(embed_model, Xte_txt)

    # 2) Train classifiers on embeddings
    models = {
        "NB(Gaussian)": GaussianNB(),  
        "LogReg": LogisticRegression(max_iter=4000),
        "LinearSVM": LinearSVC(),
        "RF": RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, clf in models.items():
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        m = eval_metrics(yte, pred)
        rows.append({"part": "embeddings", "embed_model": embed_model, "model": name, **m})

        cm = confusion_mat(yte, pred)
        top_confusions(cm, label_names, top_k=12).to_csv(
            f"{output_dir}/top_confusions_part2_{name}.csv", index=False
        )

    df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    df.to_csv(f"{output_dir}/metrics_part2.csv", index=False)
    print(df)


if __name__ == "__main__":
    run()