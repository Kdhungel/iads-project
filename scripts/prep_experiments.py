"""
Prepare data for the four experiments requested in the 11 Aug meeting.

Produces an 80/20 split (no temporal ordering) from the cleaned 71-column data,
plus binary labels and a variant with SQL Injection removed.

Run once. Everything downstream loads from what this writes.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

BASE = '/scratch/kdhungel/iads-project/data/processed'
OUT = f'{BASE}/exp'
os.makedirs(OUT, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------- load
# Combine the old 70/15/15 splits back together, then re-split 80/20.
X = pd.concat([
    pd.read_csv(f'{BASE}/X_train_clean.csv'),
    pd.read_csv(f'{BASE}/X_val_clean.csv'),
    pd.read_csv(f'{BASE}/X_test_clean.csv'),
], ignore_index=True)

y = pd.concat([
    pd.read_csv(f'{BASE}/y_train_clean.csv').squeeze(),
    pd.read_csv(f'{BASE}/y_val_clean.csv').squeeze(),
    pd.read_csv(f'{BASE}/y_test_clean.csv').squeeze(),
], ignore_index=True)

y = y.str.replace('Web Attack \ufffd', 'Web Attack -', regex=False)

print(f"Combined: {X.shape[0]:,} rows, {X.shape[1]} features")
print("\nClass distribution:")
print(y.value_counts())

# ---------------------------------------------------------------- 80/20 split
# Stratified so each class keeps its proportion. No chronological ordering,
# as instructed in the meeting.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

print(f"\nTrain: {X_tr.shape[0]:,}   Test: {X_te.shape[0]:,}")

# ---------------------------------------------------------------- scale
# Fit on train only.
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)
joblib.dump(scaler, f'{OUT}/scaler_8020.pkl')

np.save(f'{OUT}/X_train_scaled.npy', X_tr_s)
np.save(f'{OUT}/X_test_scaled.npy', X_te_s)
X_tr.to_csv(f'{OUT}/X_train_raw.csv', index=False)   # for the tree models
X_te.to_csv(f'{OUT}/X_test_raw.csv', index=False)
y_tr.to_csv(f'{OUT}/y_train.csv', index=False)
y_te.to_csv(f'{OUT}/y_test.csv', index=False)

# ---------------------------------------------------------------- EXP 2/4: binary
# Everything that is not BENIGN becomes ATTACK.
y_tr_bin = (y_tr != 'BENIGN').astype(int)
y_te_bin = (y_te != 'BENIGN').astype(int)

print(f"\nBinary — train: {y_tr_bin.sum():,} attack / {(1-y_tr_bin).sum():,} benign")
print(f"Binary — test:  {y_te_bin.sum():,} attack / {(1-y_te_bin).sum():,} benign")
print(f"Attack proportion: {y_tr_bin.mean()*100:.2f}%")

y_tr_bin.to_csv(f'{OUT}/y_train_binary.csv', index=False)
y_te_bin.to_csv(f'{OUT}/y_test_binary.csv', index=False)

# ---------------------------------------------------------------- EXP 3: drop SQL Injection
mask_tr = y_tr != 'Web Attack - Sql Injection'
mask_te = y_te != 'Web Attack - Sql Injection'

np.save(f'{OUT}/X_train_scaled_nosql.npy', X_tr_s[mask_tr.values])
np.save(f'{OUT}/X_test_scaled_nosql.npy',  X_te_s[mask_te.values])
y_tr[mask_tr].to_csv(f'{OUT}/y_train_nosql.csv', index=False)
y_te[mask_te].to_csv(f'{OUT}/y_test_nosql.csv', index=False)

print(f"\nSQL Injection removed: {(~mask_tr).sum()} train rows, {(~mask_te).sum()} test rows")
print(f"Remaining classes: {y_tr[mask_tr].nunique()}")

# ---------------------------------------------------------------- SMOTE for the deep models
# Multiclass version, minority classes to 10,000. Training set only.
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr)

strategy = {i: 10000 for i in range(len(le.classes_))
            if np.sum(y_tr_enc == i) < 10000}
sm = SMOTE(sampling_strategy=strategy, random_state=SEED)
X_res, y_res = sm.fit_resample(X_tr_s, y_tr_enc)
np.save(f'{OUT}/X_train_smote.npy', X_res)
np.save(f'{OUT}/y_train_smote.npy', y_res)
print(f"\nSMOTE multiclass: {len(X_res):,} rows")

# Binary version. Attacks are already ~20% so this is much milder.
sm_bin = SMOTE(random_state=SEED)
X_res_b, y_res_b = sm_bin.fit_resample(X_tr_s, y_tr_bin.values)
np.save(f'{OUT}/X_train_smote_binary.npy', X_res_b)
np.save(f'{OUT}/y_train_smote_binary.npy', y_res_b)
print(f"SMOTE binary: {len(X_res_b):,} rows")

# No-SQL version.
le_ns = LabelEncoder()
y_tr_ns = le_ns.fit_transform(y_tr[mask_tr])
strategy_ns = {i: 10000 for i in range(len(le_ns.classes_))
               if np.sum(y_tr_ns == i) < 10000}
sm_ns = SMOTE(sampling_strategy=strategy_ns, random_state=SEED)
X_res_ns, y_res_ns = sm_ns.fit_resample(X_tr_s[mask_tr.values], y_tr_ns)
np.save(f'{OUT}/X_train_smote_nosql.npy', X_res_ns)
np.save(f'{OUT}/y_train_smote_nosql.npy', y_res_ns)
print(f"SMOTE no-SQL: {len(X_res_ns):,} rows")

print(f"\nAll files written to {OUT}")