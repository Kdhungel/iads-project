"""
Closes the outstanding classical-model work.
  Rahman point 7 - XGBoost and RF trained WITH SMOTE, breaking the confound
  Rahman point 6 - three seeds for every classical model
  Proposal C3    - AUC-ROC, False Positive Rate, McNemar's test
CPU only, roughly 20 minutes.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import statistics as st
import json, time

D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'
SEEDS = [42, 123, 456]

X_tr_raw = pd.read_csv(f'{D}/X_train_raw.csv')
X_te_raw = pd.read_csv(f'{D}/X_test_raw.csv')
X_tr_s   = np.load(f'{D}/X_train_scaled.npy')
X_te_s   = np.load(f'{D}/X_test_scaled.npy')

y_tr_multi = pd.read_csv(f'{D}/y_train.csv').squeeze()
y_te_multi = pd.read_csv(f'{D}/y_test.csv').squeeze()
y_tr_bin   = pd.read_csv(f'{D}/y_train_binary.csv').squeeze().values
y_te_bin   = pd.read_csv(f'{D}/y_test_binary.csv').squeeze().values

le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr_multi)
y_te_enc = le.transform(y_te_multi)
CLASSES = list(le.classes_)

print(f"Train {X_tr_raw.shape}   Test {X_te_raw.shape}")
print(f"Classes {len(CLASSES)}   Binary attack rate {y_tr_bin.mean()*100:.2f}%\n")

results = {}
preds_store = {}

def false_positive_rate(cm, idx):
    fp = cm[:, idx].sum() - cm[idx, idx]
    tn = cm.sum() - cm[idx, :].sum() - cm[:, idx].sum() + cm[idx, idx]
    return fp / (fp + tn) if (fp + tn) else 0.0

def evaluate(name, model, Xa, ya, Xb, yb, names, tag, multiclass):
    t0 = time.time()
    model.fit(Xa, ya)
    train_t = time.time() - t0
    t1 = time.time()
    pred = model.predict(Xb)
    infer_t = time.time() - t1
    proba = model.predict_proba(Xb)
    cm = confusion_matrix(yb, pred)
    try:
        if multiclass:
            auc = roc_auc_score(yb, proba, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(yb, proba[:, 1])
    except Exception as e:
        auc = float('nan')
        print(f"  AUC failed: {e}")
    fprs = {names[i]: false_positive_rate(cm, i) for i in range(len(names))}
    rec = {'macro_f1': f1_score(yb, pred, average='macro', zero_division=0),
           'accuracy': accuracy_score(yb, pred), 'auc_roc': auc,
           'train_s': train_t, 'infer_s': infer_t, 'fpr': fprs}
    results[tag] = rec
    preds_store[tag] = pred
    print(f"{name}")
    print(f"  macro F1 {rec['macro_f1']:.4f}   acc {rec['accuracy']*100:.2f}%   "
          f"AUC {auc:.4f}   train {train_t:.1f}s")
    with open(f'{R}/tier1_{tag}_report.txt', 'w') as f:
        f.write(f"{name}\n")
        f.write(f"macro_f1={rec['macro_f1']:.4f} accuracy={rec['accuracy']:.4f} auc_roc={auc:.4f}\n")
        f.write(f"train_time_s={train_t:.1f} inference_time_s={infer_t:.2f}\n\n")
        f.write(classification_report(yb, pred, target_names=names, digits=4, zero_division=0))
        f.write("\n\nPer-class false positive rate\n")
        for k, v in fprs.items():
            f.write(f"  {k:<30}{v:.6f}\n")
        f.write("\nConfusion matrix\nlabels: " + ", ".join(names) + "\n")
        f.write(np.array2string(cm, max_line_width=250))
    return pred

def rf(seed):
    return RandomForestClassifier(n_estimators=100, max_depth=20,
                                  class_weight='balanced', random_state=seed, n_jobs=-1)
def rf_plain(seed):
    return RandomForestClassifier(n_estimators=100, max_depth=20,
                                  random_state=seed, n_jobs=-1)
def xgb(seed):
    return XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.3,
                         random_state=seed, n_jobs=-1, eval_metric='logloss', verbosity=0)

print("="*68)
print("PART 1  Multiclass, no SMOTE, three seeds")
print("="*68)
for s in SEEDS:
    evaluate(f"Random Forest seed {s}", rf(s), X_tr_raw, y_tr_enc,
             X_te_raw, y_te_enc, CLASSES, f"rf_multi_s{s}", True)
    evaluate(f"XGBoost seed {s}", xgb(s), X_tr_raw, y_tr_enc,
             X_te_raw, y_te_enc, CLASSES, f"xgb_multi_s{s}", True)
evaluate("Gaussian NB multiclass", GaussianNB(), X_tr_s, y_tr_enc,
         X_te_s, y_te_enc, CLASSES, "nb_multi", True)

print()
print("="*68)
print("PART 2  Multiclass WITH SMOTE  (Rahman point 7)")
print("="*68)
strategy = {i: 10000 for i in range(len(CLASSES)) if np.sum(y_tr_enc == i) < 10000}
sm = SMOTE(sampling_strategy=strategy, random_state=42)
X_sm, y_sm = sm.fit_resample(X_tr_raw, y_tr_enc)
print(f"SMOTE applied to raw features: {len(X_sm):,} rows\n")
for s in SEEDS:
    evaluate(f"Random Forest + SMOTE seed {s}", rf_plain(s), X_sm, y_sm,
             X_te_raw, y_te_enc, CLASSES, f"rf_smote_s{s}", True)
    evaluate(f"XGBoost + SMOTE seed {s}", xgb(s), X_sm, y_sm,
             X_te_raw, y_te_enc, CLASSES, f"xgb_smote_s{s}", True)

print()
print("="*68)
print("PART 3  Binary, three seeds, with and without SMOTE")
print("="*68)
sm_b = SMOTE(random_state=42)
X_sm_b, y_sm_b = sm_b.fit_resample(X_tr_raw, y_tr_bin)
print(f"Binary SMOTE: {len(X_sm_b):,} rows\n")
BN = ['BENIGN', 'ATTACK']
for s in SEEDS:
    evaluate(f"RF binary seed {s}", rf(s), X_tr_raw, y_tr_bin,
             X_te_raw, y_te_bin, BN, f"rf_bin_s{s}", False)
    evaluate(f"XGBoost binary seed {s}", xgb(s), X_tr_raw, y_tr_bin,
             X_te_raw, y_te_bin, BN, f"xgb_bin_s{s}", False)
    evaluate(f"XGBoost binary + SMOTE seed {s}", xgb(s), X_sm_b, y_sm_b,
             X_te_raw, y_te_bin, BN, f"xgb_bin_smote_s{s}", False)

print()
print("="*68)
print("PART 4  Variance across seeds")
print("="*68)
def summarise(prefix, label):
    vals = [results[f"{prefix}_s{s}"]['macro_f1'] for s in SEEDS if f"{prefix}_s{s}" in results]
    if len(vals) > 1:
        print(f"  {label:<36}{st.mean(vals):.4f} +/- {st.stdev(vals):.4f}   [{min(vals):.4f}, {max(vals):.4f}]")
summarise("rf_multi", "RF multiclass")
summarise("xgb_multi", "XGBoost multiclass")
summarise("rf_smote", "RF multiclass + SMOTE")
summarise("xgb_smote", "XGBoost multiclass + SMOTE")
summarise("rf_bin", "RF binary")
summarise("xgb_bin", "XGBoost binary")
summarise("xgb_bin_smote", "XGBoost binary + SMOTE")

print()
print("="*68)
print("PART 5  McNemar's test")
print("="*68)
print("b = A right and B wrong, c = A wrong and B right.")
print("chi2 = (|b-c| - 1)^2 / (b+c), one degree of freedom.")
print("Critical: 3.841 p=0.05, 6.635 p=0.01, 10.828 p=0.001\n")
def mcnemar(tag_a, tag_b, y_true, label_a, label_b):
    if tag_a not in preds_store or tag_b not in preds_store: return
    a = preds_store[tag_a] == y_true
    b = preds_store[tag_b] == y_true
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        print(f"  {label_a} vs {label_b}: identical predictions"); return
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    if   chi2 > 10.828: p = "p < 0.001"
    elif chi2 > 6.635:  p = "p < 0.01"
    elif chi2 > 3.841:  p = "p < 0.05"
    else:               p = "not significant"
    better = label_a if n01 > n10 else label_b
    print(f"  {label_a} vs {label_b}")
    print(f"    b={n01:,}  c={n10:,}  chi2={chi2:,.1f}  {p}  favours {better}")
mcnemar("xgb_multi_s42", "rf_multi_s42", y_te_enc, "XGBoost", "Random Forest")
mcnemar("xgb_multi_s42", "xgb_smote_s42", y_te_enc, "XGB no SMOTE", "XGB + SMOTE")
mcnemar("rf_multi_s42", "rf_smote_s42", y_te_enc, "RF weighted", "RF + SMOTE")
mcnemar("xgb_bin_s42", "rf_bin_s42", y_te_bin, "XGB binary", "RF binary")
mcnemar("xgb_bin_s42", "xgb_bin_smote_s42", y_te_bin, "XGB bin", "XGB bin + SMOTE")

with open(f'{R}/tier1_summary.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'fpr'} for k, v in results.items()}, f, indent=2)
print(f"\nAll reports written to {R}/tier1_*")
