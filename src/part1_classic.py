import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from .data import load_20newsgroups_10k, split_stratified
from .utils import eval_metrics, confusion_mat, top_confusions

def run(output_dir="outputs", vectorizer_type="tfidf", random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    X, y, label_names = load_20newsgroups_10k(random_state=random_state)
    Xtr, Xte, ytr, yte = split_stratified(X, y, random_state=random_state)

    vec = (
        TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, ngram_range=(1,2))
        if vectorizer_type == "tfidf"
        else CountVectorizer(stop_words="english", max_df=0.9, min_df=2)
    )

    models = {
        "MNB": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=3000),
        "LinearSVM": LinearSVC(),
        "RF": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("vec", vec), ("clf", clf)])
        pipe.fit(Xtr, ytr)

        pred = pipe.predict(Xte)
        m = eval_metrics(yte, pred)
        rows.append({"part": "classic", "vectorizer": vectorizer_type, "model": name, **m})

        cm = confusion_mat(yte, pred)
        top_confusions(cm, label_names, top_k=12).to_csv(
            f"{output_dir}/top_confusions_part1_{name}.csv", index=False
        )

    df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    df.to_csv(f"{output_dir}/metrics_part1.csv", index=False)
    print(df)

if __name__ == "__main__":
    run()