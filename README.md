# Bangla Fake News Detection (Naive Bayes)

## Objective
An educational project exploring fake news detection in Bangla text using Multinomial Naive Bayes, trained on the **BanFakeNews** dataset (Kaggle, ~50K articles). Beyond building a classifier, the project investigates *why* certain techniques help or don't — specifically, how document length affects the influence of class priors, and how imbalance-handling techniques interact with Naive Bayes' underlying math.

## Dataset
- **Source:** BanFakeNews (Kaggle).
- **Files used:** `Authentic-48K.csv` + `Fake-1K.csv`, concatenated into a single working set (49,977 articles).
- **Columns:** `articleID`, `domain`, `date`, `category`, `headline`, `content`, `label` (1 = authentic, 0 = fake).
- **Class distribution:** 97.4% authentic vs 2.6% fake — a severe imbalance. A model that always predicts "authentic" would already score 97.4% accuracy while being useless, which motivated the imbalance-handling experiments below.
- **Note:** the dataset also ships `LabeledAuthentic-7K.csv` and `LabeledFake-1K.csv` (a fact-checking-annotated subset with extra columns like `source`/`relation`/`F-type`). These were investigated but **not used** for training — see "Not yet done" below.

## Preprocessing
Raw `content` text was heavily noisy — initial tokenization surfaced phone numbers, IDs, and leftover HTML/CSS fragments (e.g. `0px`) as top vocabulary tokens. A custom `clean_text()` function was built to:
- Strip HTML tags, URLs, and email addresses,
- Remove digit sequences (Bangla and Latin numerals),
- Remove CSS/markup leftovers,
- Keep only Bangla Unicode characters (U+0980–U+09FF) and whitespace.

Only 6 rows became empty after cleaning and were dropped.

**Tokenization fix:** sklearn's default `CountVectorizer` token pattern is Latin-script-oriented and was fragmenting Bangla words mid-character (e.g. splitting করেছে into কর + এছে). Fixed with a custom token pattern (`r'[\u0980-\u09FF]+'`) that treats contiguous Bangla character runs as single tokens, plus `min_df=5` to drop ultra-rare hapax tokens. Verified by inspecting the most/least frequent vocabulary words before and after the fix.

## Methodology & Experiments

All experiments use a stratified 80/20 train/test split (imbalance ratio preserved in both sets) and `MultinomialNB` from scikit-learn, unless noted otherwise.

### 1. Baseline (full article content, empirical prior)
| Metric (fake class) | Score |
|---|---|
| Precision | 0.27 |
| Recall | 0.86 |
| F1 | 0.42 |

High recall, low precision — the model catches most fake articles but with many false alarms.

### 2. Manual prior adjustment (content)
`MultinomialNB.class_log_prior_` was overwritten directly (no retraining needed) to test whether shifting the class prior changes predictions.
- Prior pushed to 80/20 and even 50/50 (fake/authentic) → **negligible change** (precision 0.27→0.25, recall 0.86→0.87).
- **Why:** the decision score is `log P(class) + Σ log P(word|class)`. For long documents (hundreds of words), the summed likelihood term is in the thousands, while a prior shift is only a difference of ~1–3 in log-space — completely swamped.

### 3. Prior adjustment on headlines (short text)
To test the "document length determines prior influence" hypothesis, the same experiment was repeated using `headline` instead of `content`.
- Headline baseline: precision 0.54, recall 0.52
- Headline @ 50/50 prior: precision 0.11, recall 0.82 — a **dramatic** swing.
- **Conclusion:** prior adjustment is a length-dependent lever. It's negligible on long-form content but highly influential on short text, because the likelihood sum is much smaller relative to the prior term.
- **Decision:** the project proceeds with **content** (not headlines) as the primary signal, since headlines are often optimized for clicks/attention (by both real and fake sources) and are noisier indicators of authenticity, whereas full content is where fabricated claims and rhetorical patterns actually surface. This means prior-tuning was deliberately set aside as a lever for the main model, based on the evidence above.

### 4. Random oversampling (content)
Training data only (never touching test data) was rebalanced via oversampling the minority (fake) class with replacement, up to parity with the majority class (~38K each).

| Metric (fake class) | Baseline | Prior @ 50/50 | Oversampled |
|---|---|---|---|
| Precision | 0.27 | 0.25 | **0.47** |
| Recall | 0.86 | 0.87 | 0.82 |
| F1 | 0.42 | 0.39 | **0.60** |

Oversampling gave the best result by far. Unlike prior adjustment, it reshapes the **likelihood estimates** `P(word|class)` — the term that actually dominates decisions on long documents — rather than only nudging the prior, which explains why it worked where prior-tuning didn't.

### 5. Manual testing on external/unseen articles
Two hand-picked articles (not from the training corpus) were tested:
- A real article from **bdnews24.com** → correctly predicted Authentic (high confidence).
- A known first-person "get-rich-quick" style viral post from **earki.com** → correctly predicted Fake (high confidence).

This is a small, informal sanity check rather than a rigorous held-out evaluation (see below).

## Key takeaway
> In Naive Bayes on imbalanced text classification, the prior's influence shrinks as document length grows, because the prior is added once while likelihood terms accumulate per word. Techniques that improve **likelihood estimation** (e.g. oversampling, which sharpens per-class word-frequency estimates) outperform techniques that only adjust the **prior**, especially on long-form documents where the prior is already negligible.

## Not yet done / next steps
This is a working checkpoint, not a finished project. Remaining pieces:
- **Rigorous unseen-data evaluation:** the plan to build a proper external test set from `LabeledAuthentic-7K.csv` / `LabeledFake-1K.csv` (filtering out the ~5,755 rows that overlap with `Authentic-48K.csv`) was scoped but not executed — only two manually picked articles were tested instead.
- **From-scratch Naive Bayes implementation:** a `NaiveBayesFromScratch` class (manual priors, Laplace-smoothed likelihoods, log-space scoring) was scaffolded but not completed or validated against the sklearn baseline.
- **Additional imbalance techniques:** SMOTE and `sample_weight`-based reweighting were discussed as comparisons but not implemented.
- **Comparison visualization:** a chart plotting precision/recall/F1 across all experiments (baseline, prior adjustment, oversampling) for the portfolio writeup.
- **Production pipeline:** data versioning, a scripted `train.py`, scheduled retraining, and a model registry/metadata file.
- **UI/UX layer:** a Streamlit or Flask app with confidence visualization and word-level explainability (highlighting which words drove a prediction), with proper Bangla font rendering.

## Tech stack
Python, pandas, numpy, scikit-learn (`CountVectorizer`, `MultinomialNB`), re (regex-based text cleaning)

## How to run
1. Place `Authentic-48K.csv` and `Fake-1K.csv` in a `data/` folder.
2. Run `preprocess.ipynb` top to bottom — it loads, cleans, splits, vectorizes, and trains/evaluates each experiment in sequence.
