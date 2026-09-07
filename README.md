# Network Intrusion Detection with Random Forest and XGBoost

Comparative evaluation of two tree-based ensemble classifiers for network intrusion detection, benchmarked on the CIC-IDS2017 dataset. Completed as an MSc Cybersecurity & Technology dissertation at Northumbria University, framed around the ransomware threat to UK healthcare infrastructure.

The research question was operational rather than purely academic: **can a tree-based classifier detect malicious network flows accurately enough, and fast enough, to be useful inside a live SOC?**

---

## Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | `FILL IN` | 99.9985% |
| Precision | `FILL IN` | `FILL IN` |
| Recall | `FILL IN` | `FILL IN` |
| F1-Score | `FILL IN` | `FILL IN` |
| ROC-AUC | 1.0 | 1.0 |
| False Positive Rate | `FILL IN` | `FILL IN` |
| Training time (s) | `FILL IN` | `FILL IN` |
| Prediction latency (ms/flow) | `FILL IN` | `FILL IN` |

**XGBoost trained approximately 4x faster than Random Forest** at equivalent detection performance. In a SOC context where models are retrained regularly against evolving traffic, that difference in training cost matters more than a fractional difference in accuracy.

Per-flow prediction latency was measured deliberately. Detection accuracy is meaningless if the model cannot keep pace with live traffic volume, and this is the metric that determines whether a classifier is deployable rather than just publishable.

---

## Honest assessment of these numbers

A 99.99% accuracy figure should invite scepticism, so here is the context up front.

**The evaluation set is close to linearly separable.** This work uses the Friday-afternoon DDoS capture from CIC-IDS2017 — 225,745 labelled network flows. DDoS traffic in this dataset is separable from benign traffic by a wide margin, and near-perfect scores on this slice are a known property of the data rather than evidence of an exceptional model. A ROC-AUC of exactly 1.0 is a signal to examine the dataset, not to celebrate the classifier.

**DDoS traffic is a proxy, not the target.** The framing of this project is ransomware in healthcare, but the models are trained on DDoS flows. DDoS was used as a stand-in for ransomware-relevant network behaviour because labelled ransomware network captures at this scale are not publicly available. This is a genuine limitation, and the findings should not be read as demonstrating ransomware detection.

**Feature selection uses variance, not predictive relevance.** Where the feature count exceeds 100, the 50 highest-variance features are retained. Variance is a crude proxy for informativeness, and because selection happens before scaling, features on larger numeric scales are structurally favoured. This was a pragmatic dimensionality reduction rather than a principled one. Mutual information or recursive feature elimination would be the more defensible choice.

**What a follow-on study would need.** Multi-class evaluation across the full CIC-IDS2017 week rather than a single attack type; cross-dataset validation on UNSW-NB15 or CIC-IDS2018 to test generalisation; and evaluation against genuine ransomware traffic such as CIC-MalMem-2022.

---

## Method

1. **Load** — CIC-IDS2017 Friday-afternoon DDoS capture, 225,745 flows
2. **Preprocess** — binary relabelling to benign/attack, null handling, infinite value replacement, label encoding of categorical columns
3. **Select** — top 50 features by variance where dimensionality exceeds 100
4. **Scale** — `StandardScaler`
5. **Split** — 70/30 stratified train/test split, `random_state=42`
6. **Train** — Random Forest (100 estimators, max depth 20) and XGBoost (100 estimators, max depth 8, learning rate 0.1)
7. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, per-class metrics, FPR/FNR, and timing

---

## Running it

```bash
pip install -r requirements.txt
python final.py
```

The dataset is not included in this repository. Download CIC-IDS2017 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and place `Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv` in the same directory as the script.

Outputs are written to `REAL_CIC_IDS_RESULTS.csv` and `REAL_CIC_IDS_FEATURE_IMPORTANCE.csv`.

---

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost

---

## Context

MSc dissertation, Northumbria University London, 2026. Submitted and defended at viva September 2026.
