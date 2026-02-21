import os
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from .data import load_20newsgroups_10k
from .llm_labeler import make_prompt, label_with_openai

def embed_all(model_name, texts, batch_size=64):
    model = SentenceTransformer(model_name)
    X = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return np.asarray(X)

def elbow_inertia(X, ks=range(2, 10)):
    vals = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        vals.append((k, float(km.inertia_)))
    return vals

def closest_docs_to_centroid(X, texts, labels, centroids, cluster_id, top_n=8):
    idx = np.where(labels == cluster_id)[0]
    vecs = X[idx]
    centroid = centroids[cluster_id]
    dists = np.linalg.norm(vecs - centroid, axis=1)
    order = np.argsort(dists)[:top_n]
    return [texts[idx[i]] for i in order]

def run(output_dir="outputs", embed_model="all-MiniLM-L6-v2", K=8):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Load data
    texts, _, _ = load_20newsgroups_10k()

    # 2) Embed all docs
    X = embed_all(embed_model, texts)

    # 3) Elbow info (to justify K<10)
    inertias = elbow_inertia(X, ks=range(2, 10))
    with open(f"{output_dir}/elbow_inertia.txt", "w", encoding="utf-8") as f:
        for k, inertia in inertias:
            f.write(f"{k},{inertia}\n")
    print("Saved elbow inertias to outputs/elbow_inertia.txt")

    # 4) Top-level clustering (K must be < 10)
    km = KMeans(n_clusters=K, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    sizes = Counter(labels)
    largest_two = [cid for cid, _ in sizes.most_common(2)]

    # 5) Label top-level clusters using OpenAI
    top_labels = {}
    for cid in range(K):
        reps = closest_docs_to_centroid(X, texts, labels, km.cluster_centers_, cid, top_n=8)
        prompt = make_prompt(reps, level="top")
        top_labels[cid] = label_with_openai(prompt)

    # 6) Recluster 2 biggest clusters into exactly 3 subclusters each
    subtree = {}
    for cid in largest_two:
        idx = np.where(labels == cid)[0]
        Xsub = X[idx]
        tsub = [texts[i] for i in idx]

        km2 = KMeans(n_clusters=3, random_state=42, n_init="auto")
        sub_labels = km2.fit_predict(Xsub)

        subtree[cid] = {}
        for scid in range(3):
            sub_idx = np.where(sub_labels == scid)[0]
            vecs = Xsub[sub_idx]
            centroid = km2.cluster_centers_[scid]
            dists = np.linalg.norm(vecs - centroid, axis=1)
            order = np.argsort(dists)[:6]
            reps = [tsub[sub_idx[i]] for i in order]

            prompt = make_prompt(reps, level="sub")
            subtree[cid][scid] = label_with_openai(prompt)

    # 7) Write tree output
    lines = []
    lines.append("TOPIC TREE (partial)\n")
    for cid in range(K):
        lbl = top_labels[cid]["label"]
        desc = top_labels[cid]["description"]
        lines.append(f"- Cluster {cid} ({sizes[cid]} docs): {lbl} — {desc}")
        if cid in subtree:
            for scid in range(3):
                sl = subtree[cid][scid]["label"]
                sd = subtree[cid][scid]["description"]
                lines.append(f"  - Subcluster {cid}.{scid}: {sl} — {sd}")

    tree_text = "\n".join(lines)
    print("\n" + tree_text)

    with open(f"{output_dir}/topic_tree.txt", "w", encoding="utf-8") as f:
        f.write(tree_text)

    print("\nSaved topic tree to outputs/topic_tree.txt")

if __name__ == "__main__":
    run()