#!/usr/bin/env python3
"""
RANSOMWARE DETECTION - REAL CIC-IDS2017/2018 DATASET
Machine Learning Model Training
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_fscore_support
)
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("RANSOMWARE DETECTION - REAL CIC-IDS2017/2018 DATASET")
print("Machine Learning Model Training")
print("="*80)

# ============================================================================
# STEP 1: LOAD REAL DATASET
# ============================================================================

print("\n[1/5] Loading real CIC-IDS2017/2018 dataset...")

filenames_to_try = [
    'Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning-DDoS.pcap_ISCX.csv',
    'Thursday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Monday-WorkingHours.pcap_ISCX.csv',
]

df = None
loaded_file = None

for filename in filenames_to_try:
    try:
        print(f"  Trying to load: {filename}")
        df = pd.read_csv(filename)
        loaded_file = filename
        print(f"  ✓ Successfully loaded: {filename}")
        print(f"  ✓ Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        break
    except FileNotFoundError:
        continue
    except Exception as e:
        continue

if df is None:
    print("\n❌ ERROR: Could not find CIC-IDS CSV file!")
    print("Make sure CSV file is in the SAME folder as this script!")
    exit(1)

# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================

print("\n[2/5] Exploring dataset...")

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Data types: {(df.dtypes == 'float64').sum() + (df.dtypes == 'int64').sum()} numeric columns")

missing_count = df.isnull().sum().sum()
print(f"Missing values: {missing_count}")

label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower()]
label_col = label_cols[0] if label_cols else df.columns[-1]

print(f"\nLabel column: '{label_col}'")
print(f"Label distribution:")
print(df[label_col].value_counts())

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n[3/5] Preprocessing data...")

X = df.drop(label_col, axis=1)
y = df[label_col]

print(f"Features: {X.shape[1]} columns, {X.shape[0]:,} samples")

label_mapping = {}
for label in y.unique():
    label_str = str(label).lower()
    if 'benign' in label_str or label_str == 'unlabeled':
        label_mapping[label] = 0
    else:
        label_mapping[label] = 1

y = y.map(label_mapping)

if y.nunique() > 2:
    most_common = y.value_counts().idxmax()
    y = (y != most_common).astype(int)

print(f"Class 0 (Benign): {(y==0).sum():,}")
print(f"Class 1 (Attack): {(y==1).sum():,}")

X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

if X.shape[1] > 100:
    print(f"\nSelecting top 50 features from {X.shape[1]} total features...")
    variances = X.var()
    top_features = variances.nlargest(50).index
    X = X[top_features]

print(f"Final feature set: {X.shape[1]} features")

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Splitting data (70% train, 30% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# ============================================================================
# STEP 4: TRAIN MODELS
# ============================================================================

print("\n[4/5] Training machine learning models...\n")

results = []
feature_importance_data = None

# RANDOM FOREST
print("  Training Random Forest Classifier...")

try:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    start_train = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - start_train

    start_pred = time.time()
    y_pred_rf = rf.predict(X_test)
    rf_predict_time = time.time() - start_pred
    y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
    rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)
    rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
    rf_auc = roc_auc_score(y_test, y_pred_proba_rf) if len(y_test.unique()) > 1 else 0

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf, labels=[0, 1]).ravel()
    rf_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    rf_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    rf_prec_c, rf_rec_c, rf_f1_c, rf_support_c = precision_recall_fscore_support(
        y_test, y_pred_rf, labels=[0, 1], zero_division=0
    )

    print(f"    ✓ Accuracy:  {rf_acc:.4f}")
    print(f"    ✓ Precision: {rf_prec:.4f}")
    print(f"    ✓ Recall:    {rf_rec:.4f}")
    print(f"    ✓ F1-Score:  {rf_f1:.4f}")
    print(f"    ✓ ROC-AUC:   {rf_auc:.4f}")
    print(f"    ✓ Confusion Matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"    ✓ False Positive Rate: {rf_fpr:.6f}")
    print(f"    ✓ False Negative Rate: {rf_fnr:.6f}")
    print(f"    ✓ Per-class (Benign=0): Precision={rf_prec_c[0]:.4f}, Recall={rf_rec_c[0]:.4f}, F1={rf_f1_c[0]:.4f}")
    print(f"    ✓ Per-class (Attack=1): Precision={rf_prec_c[1]:.4f}, Recall={rf_rec_c[1]:.4f}, F1={rf_f1_c[1]:.4f}")
    print(f"    ✓ Training time:   {rf_train_time:.4f} s")
    print(f"    ✓ Prediction time: {rf_predict_time:.4f} s for {len(X_test):,} flows "
          f"({(rf_predict_time/len(X_test))*1000:.6f} ms/flow)\n")

    results.append({
        'Model': 'Random Forest',
        'Accuracy': rf_acc, 'Precision': rf_prec, 'Recall': rf_rec,
        'F1-Score': rf_f1, 'ROC-AUC': rf_auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'FPR': rf_fpr, 'FNR': rf_fnr,
        'Precision_Benign': rf_prec_c[0], 'Recall_Benign': rf_rec_c[0], 'F1_Benign': rf_f1_c[0],
        'Precision_Attack': rf_prec_c[1], 'Recall_Attack': rf_rec_c[1], 'F1_Attack': rf_f1_c[1],
        'Train_Time_s': rf_train_time, 'Predict_Time_s': rf_predict_time,
        'Predict_Time_per_flow_ms': (rf_predict_time/len(X_test))*1000
    })

    feature_importance_data = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

except Exception as e:
    print(f"    ❌ Error: {e}\n")

# XGBOOST
print("  Training XGBoost Classifier...")

try:
    xgb = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    start_train = time.time()
    xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - start_train

    start_pred = time.time()
    y_pred_xgb = xgb.predict(X_test)
    xgb_predict_time = time.time() - start_pred
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]

    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_prec = precision_score(y_test, y_pred_xgb, zero_division=0)
    xgb_rec = recall_score(y_test, y_pred_xgb, zero_division=0)
    xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
    xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb) if len(y_test.unique()) > 1 else 0

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb, labels=[0, 1]).ravel()
    xgb_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    xgb_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    xgb_prec_c, xgb_rec_c, xgb_f1_c, xgb_support_c = precision_recall_fscore_support(
        y_test, y_pred_xgb, labels=[0, 1], zero_division=0
    )

    print(f"    ✓ Accuracy:  {xgb_acc:.4f}")
    print(f"    ✓ Precision: {xgb_prec:.4f}")
    print(f"    ✓ Recall:    {xgb_rec:.4f}")
    print(f"    ✓ F1-Score:  {xgb_f1:.4f}")
    print(f"    ✓ ROC-AUC:   {xgb_auc:.4f}")
    print(f"    ✓ Confusion Matrix -> TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"    ✓ False Positive Rate: {xgb_fpr:.6f}")
    print(f"    ✓ False Negative Rate: {xgb_fnr:.6f}")
    print(f"    ✓ Per-class (Benign=0): Precision={xgb_prec_c[0]:.4f}, Recall={xgb_rec_c[0]:.4f}, F1={xgb_f1_c[0]:.4f}")
    print(f"    ✓ Per-class (Attack=1): Precision={xgb_prec_c[1]:.4f}, Recall={xgb_rec_c[1]:.4f}, F1={xgb_f1_c[1]:.4f}")
    print(f"    ✓ Training time:   {xgb_train_time:.4f} s")
    print(f"    ✓ Prediction time: {xgb_predict_time:.4f} s for {len(X_test):,} flows "
          f"({(xgb_predict_time/len(X_test))*1000:.6f} ms/flow)\n")

    results.append({
        'Model': 'XGBoost',
        'Accuracy': xgb_acc, 'Precision': xgb_prec, 'Recall': xgb_rec,
        'F1-Score': xgb_f1, 'ROC-AUC': xgb_auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
        'FPR': xgb_fpr, 'FNR': xgb_fnr,
        'Precision_Benign': xgb_prec_c[0], 'Recall_Benign': xgb_rec_c[0], 'F1_Benign': xgb_f1_c[0],
        'Precision_Attack': xgb_prec_c[1], 'Recall_Attack': xgb_rec_c[1], 'F1_Attack': xgb_f1_c[1],
        'Train_Time_s': xgb_train_time, 'Predict_Time_s': xgb_predict_time,
        'Predict_Time_per_flow_ms': (xgb_predict_time/len(X_test))*1000
    })

except Exception as e:
    print(f"    ❌ Error: {e}\n")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("[5/5] Saving results...\n")

results_df = pd.DataFrame(results)
results_df.to_csv('REAL_CIC_IDS_RESULTS.csv', index=False)
print("✓ Saved: REAL_CIC_IDS_RESULTS.csv")

if feature_importance_data is not None:
    feature_importance_data.to_csv('REAL_CIC_IDS_FEATURE_IMPORTANCE.csv', index=False)
    print("✓ Saved: REAL_CIC_IDS_FEATURE_IMPORTANCE.csv")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print(f"\nDataset: {loaded_file}")
print(f"Total samples: {df.shape[0]:,}")
print(f"Total features: {X.shape[1]}")
print(f"Training samples: {X_train.shape[0]:,}")
print(f"Test samples: {X_test.shape[0]:,}")

print("\n" + "-"*80)
print("MODEL PERFORMANCE:")
print("-"*80)
print("\n" + results_df.to_string(index=False))

print("\n" + "-"*80)
print("TOP 10 FEATURES:")
print("-"*80)
if feature_importance_data is not None:
    print("\n" + feature_importance_data.head(10).to_string(index=False))

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)

best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_accuracy = results_df.loc[best_model_idx, 'Accuracy']

print(f"\nBest Model: {best_model}")
print(f"Best Accuracy: {best_accuracy:.6f}")
print(f"Best ROC-AUC: {results_df.loc[best_model_idx, 'ROC-AUC']:.6f}")

print("\n" + "="*80)
