"""
Day-based split and duplicate statistics.
Addresses the item Dr. Rahman placed third on his priority list.
PART 1 duplicate statistics.  PART 2 day-based split on the binary task.
CPU only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import glob, os, time

RAW = '/scratch/kdhungel/iads-project/data/MachineLearningCVE'
D   = '/scratch/kdhungel/iads-project/data/processed/exp'
R   = '/scratch/kdhungel/iads-project/results'
SEED = 42

out = open(f'{R}/daysplit_report.txt', 'w')
def say(s=""):
    print(s); out.write(str(s) + "\n"); out.flush()

say("="*70)
say("PART 1  Duplicate statistics")
say("="*70)

X_tr = pd.read_csv(f'{D}/X_train_raw.csv')
X_te = pd.read_csv(f'{D}/X_test_raw.csv')
y_tr = pd.read_csv(f'{D}/y_train.csv').squeeze()
y_te = pd.read_csv(f'{D}/y_test.csv').squeeze()
say(f"Train {X_tr.shape[0]:,} rows    Test {X_te.shape[0]:,} rows")
say()

def row_hashes(df):
    arr = np.ascontiguousarray(df.values)
    return pd.util.hash_pandas_object(pd.DataFrame(arr), index=False).values

h_tr = row_hashes(X_tr); h_te = row_hashes(X_te)
dup_tr = len(h_tr) - len(np.unique(h_tr))
dup_te = len(h_te) - len(np.unique(h_te))
say(f"Duplicate rows within train:  {dup_tr:,}  ({100*dup_tr/len(h_tr):.2f}%)")
say(f"Duplicate rows within test:   {dup_te:,}  ({100*dup_te/len(h_te):.2f}%)")

set_tr = set(h_tr.tolist())
leaked_mask = np.array([h in set_tr for h in h_te])
leaked = int(leaked_mask.sum())
say(f"Test rows that also appear in train: {leaked:,}  ({100*leaked/len(h_te):.2f}%)")
say()
say("Leakage by class")
say(f"{'Class':<30}{'Test rows':>12}{'Also in train':>15}{'Percent':>10}")
say("-"*67)
for cls in sorted(y_te.unique()):
    m = (y_te.values == cls); n = int(m.sum()); lk = int(leaked_mask[m].sum())
    say(f"{cls:<30}{n:>12,}{lk:>15,}{100*lk/n if n else 0:>9.1f}%")
say()

say("="*70)
say("PART 2  Day-based split")
say("="*70)
files = sorted(glob.glob(f'{RAW}/*.csv'))
if not files:
    say(f"No CSV files at {RAW}. Skipping Part 2."); out.close(); raise SystemExit(0)
say(f"Found {len(files)} daily capture files:")
for f in files: say(f"  {os.path.basename(f)}")
say()

frames = []
for f in files:
    df = pd.read_csv(f, low_memory=False)
    df.columns = df.columns.str.strip()
    df['__day'] = os.path.basename(f)
    frames.append(df)
data = pd.concat(frames, ignore_index=True)
say(f"Merged: {data.shape[0]:,} rows")

label_col = 'Label' if 'Label' in data.columns else data.columns[-2]
data[label_col] = data[label_col].astype(str).str.replace(
    'Web Attack \ufffd', 'Web Attack -', regex=False).str.strip()

say()
say("Class presence by day")
pivot = pd.crosstab(data[label_col], data['__day'])
for cls in pivot.index:
    say(f"  {cls:<30}appears in {int((pivot.loc[cls] > 0).sum())} of {len(files)} files")
single_day = [c for c in pivot.index if (pivot.loc[c] > 0).sum() == 1]
say()
say(f"{len(single_day)} classes appear in only one capture file:")
for c in single_day: say(f"  {c}")
say()
say("These cannot survive a day-based split, so the multiclass task is not")
say("evaluable this way. The split below runs on the binary task.")
say()

drop_cols = ['Fwd PSH Flags','Bwd PSH Flags','Fwd URG Flags','Bwd URG Flags',
             'RST Flag Count','CWE Flag Count','ECE Flag Count']
feat = data.drop(columns=[c for c in drop_cols if c in data.columns])
feat = feat.drop(columns=['__day', label_col])
feat = feat.replace([np.inf, -np.inf], np.nan)
good = feat.notna().all(axis=1)
if 'Flow Duration' in feat.columns:
    good &= (feat['Flow Duration'] >= 0)
feat = feat[good]; lab = data.loc[good, label_col]; day = data.loc[good, '__day']
say(f"After cleaning: {len(feat):,} rows, {feat.shape[1]} features")

y_bin = (lab != 'BENIGN').astype(int)
days = sorted(day.unique())
n_train_days = max(1, int(round(len(days) * 0.6)))
train_days = days[:n_train_days]; test_days = days[n_train_days:]
say(); say(f"Train on {len(train_days)} files:")
for d in train_days: say(f"  {d}")
say(f"Test on {len(test_days)} files:")
for d in test_days: say(f"  {d}")

tr_m = day.isin(train_days).values; te_m = day.isin(test_days).values
Xa, ya = feat[tr_m], y_bin[tr_m]; Xb, yb = feat[te_m], y_bin[te_m]
say(); say(f"Train {Xa.shape[0]:,} rows, attack rate {ya.mean()*100:.2f}%")
say(f"Test  {Xb.shape[0]:,} rows, attack rate {yb.mean()*100:.2f}%")

if ya.nunique() < 2 or yb.nunique() < 2:
    say("One partition has a single class. Cannot evaluate."); out.close(); raise SystemExit(0)

say(); say("Duplicate check across the day-based split")
ha = row_hashes(Xa); hb = row_hashes(Xb); sa = set(ha.tolist())
lk = int(np.sum([h in sa for h in hb]))
say(f"  Test rows also in train: {lk:,} ({100*lk/len(hb):.2f}%)")
say(f"  Compare with {100*leaked/len(h_te):.2f}% under the random split.")
say()

for name, mk in [("Random Forest", lambda: RandomForestClassifier(
                    n_estimators=100, max_depth=20, class_weight='balanced',
                    random_state=SEED, n_jobs=-1)),
                 ("XGBoost", lambda: XGBClassifier(
                    n_estimators=100, max_depth=8, learning_rate=0.3,
                    random_state=SEED, n_jobs=-1, eval_metric='logloss', verbosity=0))]:
    say("-"*70); say(f"{name}, day-based split, binary task"); say("-"*70)
    m = mk(); t0 = time.time(); m.fit(Xa, ya); tt = time.time() - t0
    pred = m.predict(Xb); cm = confusion_matrix(yb, pred)
    say(classification_report(yb, pred, target_names=['BENIGN','ATTACK'],
                              digits=4, zero_division=0))
    say(f"macro F1 {f1_score(yb, pred, average='macro', zero_division=0):.4f}    train {tt:.1f}s")
    say("Confusion matrix, labels BENIGN, ATTACK")
    say(np.array2string(cm, max_line_width=200)); say()

say("="*70)
say("Compare the macro F1 above against the random-split binary results.")
say("A substantial drop indicates the random split was inflating performance")
say("through session overlap rather than measuring generalisation.")
out.close()
print(f"\nSaved {R}/daysplit_report.txt")
