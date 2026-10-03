<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mailspamchecker.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.47%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

# Email Spam Checker

**Live app:** <https://mailspamchecker.streamlit.app/>

An end-to-end **Machine Learning** and **Natural Language Processing** application that
classifies emails as **Spam** or **Ham** using TF-IDF features and a **Perceptron**
classifier, with explainable AI, batch scoring and an interactive analytics dashboard.

</div>

---

## Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Screenshots](#screenshots)
- [Dataset](#dataset)
- [How it works](#how-it-works)
- [Text preprocessing](#text-preprocessing)
- [Model](#model)
- [Performance](#performance)
- [Explainable AI](#explainable-ai)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Retraining the model](#retraining-the-model)
- [Roadmap](#roadmap)

---

## Overview

Spam filtering is a classic binary text-classification problem, and a good showcase of the
full machine-learning workflow: raw text cleaning, feature extraction, linear modelling,
evaluation, interpretability and deployment.

**Email Spam Checker** covers that entire pipeline. The trained model is served through a
six-page Streamlit dashboard that lets you classify one email or thousands, inspect exactly
which words drove a decision, and explore the vocabulary the model actually learned.

## Highlights

- **Single email prediction** with decision score, confidence band and a clear Spam/Ham verdict
- **Explainable AI** - every prediction is decomposed into per-word feature contributions
- **Batch prediction** from an uploaded CSV, scored in a single vectorised pass
- **Vocabulary explorer** - search all 10,000 learned features and inspect their weights
- **Performance dashboard** - accuracy, precision, recall, F1 and the confusion matrix, all read
  straight from the stored evaluation artefacts so the numbers can never drift apart
- **Light and dark themes** via a single sidebar toggle
- **CSV export** for both predictions and per-feature explanations

## Screenshots

### Home

![Home](images/Home_Page.png)

### Single prediction

![Single prediction](images/Single_Prediction.png)

### Batch prediction

![Batch prediction](images/Batch_Prediction.png)

## Dataset

Built from the *Spam & Ham Email Dataset* (`spam_ham_dataset.csv`), with two columns:

| Column | Description |
| --- | --- |
| `text` | Raw email body |
| `label` | `0` = Ham, `1` = Spam |

| Metric | Value |
| --- | ---: |
| Total emails | 193,812 |
| Ham emails | 102,159 |
| Spam emails | 91,653 |
| Training samples | 155,049 (80%) |
| Test samples | 38,763 (20%) |

The split is stratified (`test_size=0.20`, `random_state=42`), so class proportions are
preserved across train and test.

## How it works

```
Raw email
   |
   v
[1] Clean text ......... lowercase, strip HTML/URLs/emails, drop numbers + punctuation
   |
   v
[2] Lemmatise .......... English stopwords removed, WordNet lemmatisation
   |
   v
[3] Vectorise .......... TF-IDF, max_features = 10,000, ngram_range = (1, 2)
   |
   v
[4] Score .............. Perceptron decision function
   |
   v
[5] Explain ............ tfidf_value * model_weight per active feature
   |
   v
Spam / Ham + confidence + influential words
```

## Text preprocessing

`utils/preprocessing.py` is the shared contract between training and serving. If this changes,
the saved model must be retrained - otherwise the token stream drifts and predictions silently
degrade. The steps, in order:

1. Lowercase
2. Strip HTML tags
3. Remove URLs
4. Remove email addresses
5. Remove digits
6. Remove punctuation
7. Collapse whitespace
8. Remove English stopwords
9. **Lemmatise** with the WordNet lemmatiser

The regexes and the punctuation lookup table are compiled once at import time rather than
rebuilt per email, which matters when preprocessing tens of thousands of rows.

## Model

| Component | Configuration |
| --- | --- |
| Vectoriser | `TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))` |
| Classifier | `Perceptron(max_iter=1000, eta0=1.0, random_state=42)` |
| Features | 10,000 - unigrams and bigrams |
| Classes | `0` = Ham, `1` = Spam |

The Perceptron is a linear model, which is exactly why the explainability below works: the
decision score is the sum of `value * weight` across the active features.

## Performance

Measured on the held-out test split of 38,763 emails. Headline precision and recall are
reported for the **Spam** class, treated as the positive label.

| Metric | Score |
| --- | ---: |
| Accuracy | **97.68%** |
| Precision | **98.21%** |
| Recall | **96.86%** |
| F1 Score | **97.53%** |

### Confusion matrix

Rows are the actual class, columns the predicted class.

| | Predicted Ham | Predicted Spam | Total |
| --- | ---: | ---: | ---: |
| **Actual Ham** | 20,105 | 324 | 20,429 |
| **Actual Spam** | 576 | 17,758 | 18,334 |
| **Total** | 20,681 | 18,082 | 38,763 |

Only 324 legitimate emails are wrongly flagged, while 576 spam emails slip through.

### Per-class report

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Ham | 0.9721 | 0.9841 | 0.9781 | 20,429 |
| Spam | 0.9821 | 0.9686 | 0.9753 | 18,334 |
| Macro average | 0.9771 | 0.9764 | 0.9767 | 38,763 |
| Weighted average | 0.9768 | 0.9768 | 0.9768 | 38,763 |

These figures are loaded at runtime from `results/metrics.json` and `results/confusion_matrix.npy`
rather than hard-coded in the dashboard, so the app and the artefacts can never disagree.

## Explainable AI

Each prediction is explained by attributing the decision score to individual TF-IDF features:

```
contribution = tfidf_value * model_weight
```

- **Positive contribution** pushes the prediction toward **Spam**
- **Negative contribution** pushes it toward **Ham**
- **Magnitude** indicates how strongly that word influenced the outcome

Because the Perceptron is linear, the contributions sum to the decision score. The dashboard
shows the strongest Spam and Ham words, a full contribution table, and a CSV export.

## Project structure

```
Email-Spam-Classifier/
|-- app.py                     # Landing page and entry point
|-- pages/
|   |-- Single_Prediction.py   # Single email classification + XAI
|   |-- Batch_Prediction.py    # CSV batch scoring and analytics
|   |-- Model_Analytics.py     # Vocabulary and weight exploration
|   |-- Performance.py         # Evaluation metrics dashboard
|   `-- About.py               # Project, dataset and model reference
|-- utils/
|   |-- __init__.py
|   |-- theme.py               # Palettes, global stylesheet, UI helpers
|   |-- sidebar.py             # setup_page() bootstrap + sidebar
|   |-- model_loader.py        # Artefact loading and prediction helpers
|   |-- preprocessing.py       # Cleaning, stopwords, lemmatisation
|   |-- explainable.py         # Per-feature contribution analysis
|   `-- visualization.py       # Matplotlib charts
|-- models/
|   |-- perceptron_model.pkl   # Trained classifier
|   `-- tfidf_vectorizer.pkl   # Fitted vectoriser
|-- results/
|   |-- metrics.json           # accuracy, precision, recall, f1_score
|   `-- confusion_matrix.npy   # 2x2 test-set confusion matrix
|-- notebooks/
|   `-- Email_Spam_Classification_Perceptron.ipynb
|-- images/                    # README screenshots
|-- .devcontainer/
|   `-- devcontainer.json      # Optional VS Code / Codespaces container
|-- requirements.txt
`-- README.md
```

### Design notes

- **One stylesheet.** `utils/theme.py` owns every colour and the single global stylesheet, so the
  pages cannot drift out of sync. Pages never inline CSS.
- **One page bootstrap.** Every page opens with `setup_page(title, icon)`, which sets the page
  config, renders the sidebar and injects the stylesheet in a fixed order.
- **Artefacts, not constants.** Evaluation numbers live in `results/` and are loaded at runtime.
- **Cached work.** The model is loaded once per server process; the vocabulary tables and CSV
  parsing are cached across reruns.

## Tech stack

| Technology | Version | Purpose |
| --- | --- | --- |
| Python | 3.11+ | Language |
| Streamlit | >= 1.47 | Web application |
| scikit-learn | >= 1.7 | Machine learning |
| Pandas | >= 2.3 | Data handling |
| NumPy | >= 2.3 | Numerical computing |
| Matplotlib | >= 3.10 | Charts |
| NLTK | >= 3.9 | Stopwords and lemmatisation |
| Joblib | >= 1.5 | Model serialisation |

## Getting started

**Prerequisites:** Python 3.11 or newer.

```bash
git clone https://github.com/<your-username>/Email-Spam-Classifier.git
cd Email-Spam-Classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

It opens on <http://localhost:8501>.

### NLTK data

The app needs the `stopwords` and `wordnet` corpora. They are fetched automatically on first
import, which requires network access once. To install them up front:

```bash
python -m nltk.downloader stopwords wordnet
```

If they cannot be downloaded, the app fails fast with an explicit error rather than quietly
changing the token stream.

### Optional: dev container

`.devcontainer/devcontainer.json` pins a Python 3.11 container, installs dependencies and
auto-launches Streamlit on port 8501. Use **Reopen in Container** in VS Code, or open the repo
in GitHub Codespaces. It is entirely optional - running locally needs nothing extra.

## Retraining the model

1. Put `spam_ham_dataset.csv` beside the notebook
2. Open `notebooks/Email_Spam_Classification_Perceptron.ipynb` and run every cell
3. Move the outputs into place:
   - `perceptron_model.pkl` and `tfidf_vectorizer.pkl` -> `models/`
   - `accuracy` / `precision` / `recall` / `f1_score` -> `results/metrics.json`
   - `confusion_matrix.npy` -> `results/`

Keep `utils/preprocessing.py` in sync with the notebook's cleaning step, then regenerate the
Performance dashboard figures so the app and the artefacts stay consistent.

## Roadmap

- Compare against Logistic Regression, Naive Bayes, SVM and Random Forest
- Explore LSTM or Transformer models
- Multilingual spam detection
- REST API for real-time inference
- Continuous retraining on newly labelled data
- Docker and cloud deployment

---

<div align="center">

Built with **Python**, **scikit-learn**, **Streamlit**, **TF-IDF** and **Perceptron**.

If this repository helped you learn about machine learning, NLP or Streamlit,
please consider leaving a star.

</div>
