# Gene Expression Classifier

A machine learning pipeline for classifying lung cancer subtypes from microarray gene expression data. Built on the public GEO dataset [GSE10245](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10245), this project distinguishes **adenocarcinoma (AC)** from **squamous cell carcinoma (SCC)** in non-small cell lung cancer (NSCLC) patients.

---

## Background

Accurate molecular subtyping of NSCLC is clinically important: AC and SCC carry different prognoses and respond differently to targeted therapies. This project demonstrates a reproducible bioinformatics ML workflow — from raw GEO data download through preprocessing, feature selection, and classification — using tools standard in computational biology research.

---

## Dataset

| Property | Value |
|---|---|
| GEO Accession | GSE10245 |
| Platform | Affymetrix Human Genome U133 Plus 2.0 (GPL570) |
| Samples | 58 NSCLC tumour biopsies |
| Classes | Adenocarcinoma (n=40), Squamous cell carcinoma (n=18) |
| Features | 54,675 probe sets → top 2,000 by variance |

---

## Methods

```
Raw GEO data
    └── fetch_data.py          Download .soft.gz, build expression matrix & metadata CSVs
    └── preprocess.py          Variance filter → top-k genes, optional log1p, label encoding
    └── 01_download_and_QC.ipynb   Exploratory data analysis and quality checks
    └── 02_train.ipynb         Model training, cross-validation, evaluation
```

**Pipeline steps:**
1. **Download** — `GEOparse` fetches the soft file and assembles a samples × probes expression matrix alongside sample metadata.
2. **Preprocessing** — samples with mismatched metadata are dropped; top 2,000 most variable probes are selected; labels are mapped from free-text characteristics to binary integers.
3. **Train/test split** — stratified 80/20 split; `StandardScaler` is fit exclusively on the training set to prevent data leakage.
4. **Models** — Logistic Regression (L2) and Random Forest (200 trees), each evaluated with 5-fold stratified cross-validation on the training set and a held-out test set.

---

## Results

| Model | CV Accuracy (mean ± std) | Test Accuracy |
|---|---|---|
| Logistic Regression | see notebook | see notebook |
| Random Forest | see notebook | see notebook |

> Run `02_train.ipynb` to reproduce results. Figures are saved to `figures/`.

---

## Project Structure

```
gene-expression-classifier/
├── notebooks/
│   ├── 01_download_and_QC.ipynb   # Data download, inspection, and QC
│   └── 02_train.ipynb             # Model training and evaluation
├── src/
│   ├── fetch_data.py              # GEO download and matrix construction
│   └── preprocess.py              # Feature selection and label encoding
├── data/                          # Not tracked by git (see .gitignore)
├── figures/                       # Not tracked by git (generated outputs)
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/gene-expression-classifier.git
cd gene-expression-classifier
```

**2. Create and activate a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the notebooks in order**
```bash
jupyter notebook notebooks/01_download_and_QC.ipynb
jupyter notebook notebooks/02_train.ipynb
```

The first notebook downloads ~150 MB of data from NCBI GEO into `data/`. An internet connection is required for that step.

---

## Dependencies

See `requirements.txt`. Key libraries: `GEOparse`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `umap-learn`, `shap`.

---

## License

This project uses publicly available data from NCBI GEO. Code is released under the MIT License.
