# Network Intrusion Detection with Random Forest and XGBoost

Comparative evaluation of two tree-based ensemble classifiers for network intrusion detection, benchmarked on the CIC-IDS2017 dataset. Completed as an MSc Cybersecurity & Technology dissertation at Northumbria University, framed around the ransomware threat to UK healthcare infrastructure.

The research question was operational rather than purely academic: **can a tree-based classifier detect malicious network flows accurately enough, and fast enough, to be useful inside a live SOC?**

---

## Results

Test set: 67,724 flows (30% stratified holdout from 225,745 total).

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 99.9956% | 99.9985% |
| Precision | 99.9974% | 100% |
| Recall | 99.9948% | 99.9974% |
| F1-Score | 99.9961% | 99.9987% |
| ROC-AUC | 1.0000 | 1.0000 |
| False Positive Rate | 0.0034% | 0% |
| False Negative Rate | 0.0052% | 0.0026% |
| Training time | 6.51 s | **1.72 s** |
| Prediction time (67,724 flows) | 0.156 s | **0.031 s** |
| Per-flow latency | 0.0023 ms | **0.00045 ms** |

### Confusion matrices

**Random Forest**

| | Predicted Benign | Predicted Attack |
|---|---|---|
| **Actual Benign** | TN = 29,315 | FP = 1 |
| **Actual Attack** | FN = 2 | TP = 38,406 |

**XGBoost**

| | Predicted Benign | Predicted Attack |
|---|---|---|
| **Actual Benign** | TN = 29,316 | FP = 0 |
| **Actual Attack** | FN = 1 | TP = 38,407 |

### What actually separates the two models

Detection performance is effectively tied. Across 67,724 test flows, Random Forest made 3 errors and XGBoost made 1. That difference is not statistically meaningful on a single split, and neither model can be called better on accuracy grounds.

The real difference is cost:

- **XGBoost trained 3.8x faster** — 1.72 s against 6.51 s
- **XGBoost predicted 5.0x faster** — 0.031 s against 0.156 s for the same 67,724 flows

Translated into throughput, XGBoost sustains roughly **2.2 million flows per second** against Random Forest's 434,000. Both are far above what a mid-sized network generates, so both are deployable — but in a SOC where models are retrained regularly against drifting traffic, a 4x reduction in training cost compounds across every retraining cycle.

**Zero false positives matters more than the accuracy figure.** In a SOC, false positives are the operational bottleneck: every one consumes analyst time. Random Forest produced 1 false positive out of 29,316 benign flows; XGBoost produced none. At production traffic volumes that difference scales into real analyst hours.

---

## Honest assessment of these numbers

Near-perfect scores should invite scepticism, so here is the context up front.

**The evaluation set is close to linearly separable.** This work uses the Friday-afternoon DDoS capture from CIC-IDS2017. DDoS traffic in this dataset is separable from benign traffic by a very wide margin, and near-perfect scores on this slice are a known property of the data rather than evidence of an exceptional model. A ROC-AUC of exactly 1.0000 for both models is a signal to examine the dataset, not to celebrate the classifiers.

**Three errors in 67,724 samples means the task was too easy to discriminate between models.** A more demanding benchmark — rarer attack classes, or a multi-class problem — would be needed to say anything meaningful about relative detection quality. This study can only speak to relative computational cost.

**DDoS traffic is a proxy, not the target.** The framing is ransomware in healthcare, but the models are trained on DDoS flows. DDoS was used as a stand-in for ransomware-relevant network behaviour because labelled ransomware network captures at this scale are not publicly available. These findings should not be read as demonstrating ransomware detection.

**Feature selection uses variance, not predictive relevance.** Where the feature count exceeds 100, the 50 highest-variance features are retained. Variance is a crude proxy for informativeness, and because selection happens before scaling, features on larger numeric scales are structurally favoured. This was a pragmatic dimensionality reduction rather than a principled one. Mutual information or recursive feature elimination would be more defensible.

**No cross-validation.** Results come from a single 70/30 stratified split with a fixed random seed. Repeated k-fold cross-validation would give confidence intervals and show whether the 3-versus-1 error difference is signal or noise. It almost certainly is noise.

**What a follow-on study would need.** Multi-class evaluation across the full CIC-IDS2017 week rather than a single attack type; cross-dataset validation on UNSW-NB15 or CIC-IDS2018 to test generalisation; and evaluation against genuine ransomware traffic such as CIC-MalMem-2022.

---

## Method

1. **Load** — CIC-IDS2017 Friday-afternoon DDoS capture, 225,745 flows
2. **Preprocess** — binary relabelling to benign/attack, null handling, infinite value replacement, label encoding of categorical columns
3. **Select** — top 50 features by variance where dimensionality exceeds 100
4. **Scale** — `StandardScaler`
5. **Split** — 70/30 stratified train/test split, `random_state=42`
6. **Train** — Random Forest (100 estimators, max depth 20) and XGBoost (100 estimators, max depth 8, learning rate 0.1)
7. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, per-class metrics, FPR/FNR, and training and inference timing

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
