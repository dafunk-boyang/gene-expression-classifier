import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#src/preprocess.py
def select_label_column(meta:pd.DataFrame, candidates=("char0_disease state","char0+disease state", "char0", "disease_state")):
    for c in candidates:
        if c in meta.columns:
            return c
    raise ValueError(f"Can't find column in meta among {candidates}")

def build_dataset(expr_csv, meta_csv, label_col=None, top_k=2000, log_transform=False):
    X = pd.read_csv(expr_csv, index_col=0)
    meta = pd.read_csv(meta_csv, index_col=0)

    #Ensure columns/samples match
    common_samples = X.columns.intersection(meta.index)
    X = X[common_samples]
    meta = meta.loc[common_samples]

    #Optional log transform
    if log_transform:
        X = np.log1p(X)

    #Variance filter
    variances = X.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(top_k).index
    X = X.loc[top_genes]

    #Transpose to samples x features
    X = X.T

    #Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    #Labels
    if label_col is None:
        label_col = select_label_column(meta)

    y_raw = meta[label_col].copy().astype(str).str.lower().str.strip()

    #Try Mapping to binary (Adjust as needed)
    # E.g. "tumor" -> 1, "normal" -> 0
    mapping = {
        "normal": 0, "control": 0, "benign": 0, "no": 0, "negative": 0, "squamous cell carcinoma": 0,
        "tumor": 1, "cancer": 1, "case": 1, "yes": 1, "positive": 1, "adenocarcinoma": 1
    }

    y = y_raw.map(mapping)
    valid = ~y.isna()
    X_scaled = X_scaled.loc[valid]
    y = y.loc[valid].astype(int)

    return X_scaled, y