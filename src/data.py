from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

def load_20newsgroups_10k(random_state: int = 42, n_samples: int = 10_000):
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=random_state,
    )
    X = data.data[:n_samples]
    y = data.target[:n_samples]
    return X, y, data.target_names

def split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)