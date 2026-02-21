import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def eval_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }

def confusion_mat(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def top_confusions(cm, label_names, top_k=12):
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    pairs = []
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if cm2[i, j] > 0:
                pairs.append((cm2[i, j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    rows = []
    for count, i, j in pairs[:top_k]:
        rows.append({"count": int(count), "true": label_names[i], "pred": label_names[j]})
    return pd.DataFrame(rows)