"""
Experiment 4 from the 11 Aug meeting.

Random Forest, Naive Bayes and XGBoost on the binary attack-vs-benign task,
for comparison against the transformer from experiment 2.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib, time

SEED = 42
D = '/scratch/kdhungel/iads-project/data/processed/exp'
R = '/scratch/kdhungel/iads-project/results'
M = '/scratch/kdhungel/iads-project/models'

X_tr = pd.read_csv(f'{D}/X_train_raw.csv')
X_te = pd.read_csv(f'{D}/X_test_raw.csv')
y_tr = pd.read_csv(f'{D}/y_train_binary.csv').squeeze().values
y_te = pd.read_csv(f'{D}/y_test_binary.csv').squeeze().values

X_tr_s = np.load(f'{D}/X_train_scaled.npy')
X_te_s = np.load(f'{D}/X_test_scaled.npy')

NAMES = ['BENIGN', 'ATTACK']

print(f"Train: {X_tr.shape}   Test: {X_te.shape}")
print(f"Attack proportion — train {y_tr.mean()*100:.2f}%, test {y_te.mean()*100:.2f}%\n")

results = {}

def run(name, model, Xa, Xb, tag):
    print(f"{'='*60}\n{name}\n{'='*60}")
    t0 = time.time()
    model.fit(Xa, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    pred = model.predict(Xb)
    infer_time = time.time() - t1

    rep = classification_report(y_te, pred, target_names=NAMES, digits=4)
    cm = confusion_matrix(y_te, pred)

    print(f"Train: {train_time:.1f} s   Inference: {infer_time:.2f} s "
          f"({1000*infer_time/len(y_te):.4f} ms per sample)")
    print(rep)
    print("Confusion matrix:")
    print(cm)

    with open(f'{R}/exp4_{tag}_report.txt', 'w') as f:
        f.write(f"{name} — binary attack vs benign, seed={SEED}\n")
        f.write(f"train_time_s={train_time:.1f}  inference_time_s={infer_time:.2f}\n\n")
        f.write(rep)
        f.write("\n\nConfusion matrix\nlabels: BENIGN, ATTACK\n")
        f.write(np.array2string(cm, max_line_width=200))

    joblib.dump(model, f'{M}/exp4_{tag}.pkl')
    results[name] = (train_time, infer_time)
    print()

run("Random Forest",
    RandomForestClassifier(n_estimators=100, max_depth=20,
                           class_weight='balanced', random_state=SEED, n_jobs=-1),
    X_tr, X_te, 'rf')

run("Gaussian Naive Bayes",
    GaussianNB(),
    X_tr_s, X_te_s, 'nb')

run("XGBoost",
    XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.3,
                  random_state=SEED, n_jobs=-1, eval_metric='logloss',
                  verbosity=0),
    X_tr, X_te, 'xgb')

print(f"{'='*60}\nTraining times")
for k, (tt, it) in results.items():
    print(f"  {k:<22} {tt:>8.1f} s train   {it:>6.2f} s inference")
print(f"\nAll reports saved to {R}/exp4_*_report.txt")
