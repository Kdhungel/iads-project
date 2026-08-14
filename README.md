# IAA-Transformer

**Imbalance-aware attention for network intrusion detection on CICIDS-2017.**
A negative result.

MSc Computational Sciences, Laurentian University. Supervised by Dr. Kalpdrum Passi.
Reviewed by Dr. SK Md Mizanur Rahman.

---

## What this project tested

CICIDS-2017 is severely imbalanced. BENIGN traffic is 80.3% of the dataset and the
rarest attack class has eleven samples out of 2.8 million flows. A model can score 80%
accuracy by predicting BENIGN for everything and detect nothing at all.

Standard Transformer attention has no representation of class frequency. It weights a
Heartbleed flow identically to a BENIGN one. The hypothesis was that making attention
frequency-aware would improve rare-class detection, by adding a bias term to the
attention scores:

```
IAA(Q, K, V) = softmax( QKᵀ / √d  +  α · log(1/f_c) ) · V
```

where `f_c` is the class frequency and `α` is learnable. At `α = 0` the term vanishes,
which gives the ablation baseline.

## The hypothesis does not hold

An ablation with `α` hard-fixed at zero, run across three seeds on an identical code
path, matched or outperformed the learned model on every seed. Macro F1 differences of
−0.059, −0.032 and −0.021. The frequency bias term contributes nothing.

Three further corrections emerged while checking the work.

**The first implementation did nothing at all.** The bias was shaped `[batch, 1, 1, 1]`
and broadcast onto a `[batch, heads, 71, 71]` score matrix, so every key position within
a row received the same value. Softmax is invariant to a constant added along the
dimension it normalises over, so it cancelled. Versions 1 through 4 were mathematically
inert.

**An apparent architectural gain was a bug.** The custom implementation beat PyTorch's
`nn.TransformerEncoderLayer` by 0.122 macro F1. Diffing against the PyTorch source
showed the custom attention module applied no dropout to the softmax output, where
`nn.MultiheadAttention` applies it by default. Restoring it accounted for 77% of the gap.

**The headline XSS result did not survive precision.** An earlier draft reported XSS
recall rising from 0.58 to 0.95. Including precision reverses the reading: it falls from
0.096 to 0.078 and F1 falls with it, from 0.165 to 0.144. The model roughly doubled its
false positives to gain 36 detections, and Brute Force F1 fell from 0.280 to 0.169 in the
same run.

## Results

Macro F1, three seeds each, 80/20 split.

| Model | Macro F1 | SD | Train time |
|---|---|---|---|
| **Random Forest + SMOTE** | **0.8985** | 0.0037 | 96 s |
| 1D-CNN | 0.8216 | 0.0064 | 160 s |
| Random Forest | 0.8194 | 0.0028 | 84 s |
| BiLSTM | 0.7141 | 0.0066 | 566 s |
| Transformer | 0.6315 | not replicated | 2,250 s |
| XGBoost | 0.4856 | 0.0000 | 84 s |

The CNN and Random Forest are statistically indistinguishable. The deficit is specific to
the Transformer, not general to deep learning.

## Two findings that were not anticipated

**A reproducibility failure in the project's own headline number.** An earlier draft
reported XGBoost at macro F1 0.867. Re-running the identical configuration with the same
seed returns 0.6925 on the same split. The entire difference sits in three classes
holding two, five and three test samples. Random Forest reproduces to within 0.003;
XGBoost does not. The 0.867 was itself an instance of the rare-class instability this
report warns about.

**Performance on CICIDS-2017 is substantially an artefact of partitioning.** Under a
random split, 13.4% of test flows are byte-identical to a flow in training, rising to
58.8% for PortScan and 47.0% for SSH-Patator. Replacing it with a day-based split
collapses binary macro F1 from 0.998 to between 0.45 and 0.53, with attack recall falling
from above 0.999 to below 0.09. Random Forest detects 2,043 of 267,735 attacks in
held-out days.

Part of that collapse is caused by attack families appearing on only one capture day, so
the two effects cannot be fully separated. That limitation is itself a finding: CICIDS-2017
cannot support a clean temporal evaluation. Since nearly all published work on this
dataset uses a random split, the near-perfect accuracy commonly reported measures
something closer to recognition of memorised sessions than detection of unseen traffic.

## What was tested and ruled out

| Intervention | Outcome |
|---|---|
| Frequency bias inside attention | Mathematically inert, cancelled by softmax |
| Frequency bias on output logits | Well defined, contributes nothing |
| Multi-pass inference (from the proposal) | Changes at most 0.031% of predictions |
| Class-weighted cross entropy | Training collapsed, or recall up and F1 down |
| LDAM-DRW | Recall up, precision down further, F1 lower |
| SMOTE | The only intervention that improved F1 |
| Leaky ReLU | Worse than ReLU on 14 of 15 classes |
| Dropping SQL Injection | No measurable effect on remaining classes |

## Repository

```
notebooks/    01-15, the original development sequence
scripts/      Narval HPC training and submission scripts
results/      classification reports and confusion matrices for every run
IAA_Transformer_Report.pdf    full report, 28 pages
```

Data and model weights are excluded from version control. CICIDS-2017 is available from
the Canadian Institute for Cybersecurity.

Fifty-nine training runs were carried out on the Narval cluster, Digital Research
Alliance of Canada, under allocation def-kpassi.

## Setup

```bash
conda create -n iads python=3.11
conda activate iads
pip install torch numpy pandas scikit-learn imbalanced-learn xgboost jupyterlab
```

Note that XGBoost results are version-sensitive on the rare classes; see the
reproducibility discussion above. The reported figures use XGBoost 3.2.0.

## Limitations

The Transformer was never hyperparameter-tuned and is reported at a single seed, so its
deficit relative to the CNN may partly reflect configuration rather than architecture.
SMOTE at 900-fold interpolation for Heartbleed produces near-duplicates rather than new
information. Rows containing NaN or infinite values were removed from all splits
including test, which makes the test distribution cleaner than production. CICIDS-2017
contains documented labelling errors (Engelen et al., 2021) that fall hardest on exactly
the rare classes this work concerns.

SHAP explainability and secondary-dataset validation were specified in the proposal and
not completed.
