import pandas as pd, numpy as np, time, xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score

print("xgboost version:", xgboost.__version__)
D='/scratch/kdhungel/iads-project/data/processed'
Xtr=pd.read_csv(f'{D}/X_train_clean.csv')
Xva=pd.read_csv(f'{D}/X_val_clean.csv')
ytr=pd.read_csv(f'{D}/y_train_clean.csv').squeeze().str.replace('Web Attack \ufffd','Web Attack -',regex=False)
yva=pd.read_csv(f'{D}/y_val_clean.csv').squeeze().str.replace('Web Attack \ufffd','Web Attack -',regex=False)

le=LabelEncoder(); a=le.fit_transform(ytr); b=le.transform(yva)
print("train",Xtr.shape,"val",Xva.shape, flush=True)

m=XGBClassifier(n_estimators=100,max_depth=8,learning_rate=0.3,
                random_state=42,n_jobs=-1,eval_metric='logloss',verbosity=0)
t=time.time(); m.fit(Xtr,a); print(f"trained in {time.time()-t:.1f}s", flush=True)
p=m.predict(Xva)
f1=f1_score(b,p,average='macro',zero_division=0)
print()
print("="*60)
print(f"macro F1 on ORIGINAL 70/15/15 split: {f1:.4f}")
print("REPORT CLAIMS: 0.867")
print(f"tier1 on new 80/20 split gave:      0.4856")
print("="*60)
print()
print(classification_report(b,p,target_names=le.classes_,digits=4,zero_division=0))
