import pandas as pd
import numpy as np


# Default mapping for general tumor-vs-normal datasets.
# For subtype-vs-subtype tasks (e.g. adenocarcinoma vs squamous cell carcinoma),
# pass a custom mapping: e.g. {"adenocarcinoma": 1, "squamous cell carcinoma": 0}
DEFAULT_MAPPING = {
    "normal": 0, "control": 0, "benign": 0, "no": 0, "negative": 0,
    "tumor": 1, "cancer": 1, "case": 1, "yes": 1, "positive": 1,
}


def select_label_column(meta: pd.DataFrame, candidates=("char0+disease state", "char0", "disease_state")):
    """Return the first column name from *candidates* that exists in *meta*.

    Raises ValueError if none of the candidates are found.
    """
    for c in candidates:
        if c in meta.columns:
            return c
    raise ValueError(f"Can't find label column in meta among {candidates}")


def build_dataset(
    expr_csv: str,
    meta_csv: str,
    label_col: str = None,
    top_k: int = 2000,
    log_transform: bool = False,
    mapping: dict = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load expression and metadata CSVs and return a (X, y) pair ready for ML.

    Parameters
    ----------
    expr_csv : str
        Path to the expression matrix CSV (probes × samples).
    meta_csv : str
        Path to the sample metadata CSV (samples × attributes).
    label_col : str, optional
        Column in *meta_csv* to use as the class label. Auto-detected when None.
    top_k : int
        Number of most-variable probes to retain. Default 2000.
    log_transform : bool
        Apply log1p transformation before variance filtering. Default False.
    mapping : dict, optional
        Map raw label strings to integer class codes. Falls back to
        DEFAULT_MAPPING when None.

    Returns
    -------
    X : pd.DataFrame, shape (n_samples, top_k)
        Feature matrix (samples × probes), z-score normalised across samples.
    y : pd.Series, shape (n_samples,)
        Integer class labels aligned with X.
    """
    X = pd.read_csv(expr_csv, index_col=0)
    meta = pd.read_csv(meta_csv, index_col=0)

    # Ensure columns/samples match
    common_samples = X.columns.intersection(meta.index)
    X = X[common_samples]
    meta = meta.loc[common_samples]

    # Optional log transform
    if log_transform:
        X = np.log1p(X)

    # Variance filter — keep top-k most variable probes
    variances = X.var(axis=1)
    top_genes = variances.sort_values(ascending=False).head(top_k).index
    X = X.loc[top_genes]

    # Transpose to samples × features
    X = X.T

    # Labels
    if label_col is None:
        label_col = select_label_column(meta)

    y_raw = meta[label_col].copy().astype(str).str.lower().str.strip()

    label_mapping = mapping if mapping is not None else DEFAULT_MAPPING
    y = y_raw.map(label_mapping)
    valid = ~y.isna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    return X, y
