```markdown
# IAA-Transformer: Imbalance-Aware Attention Transformer for Network Intrusion Detection

## Overview

Standard Transformer attention treats every class equally during training. The attention mechanism has no awareness of class frequency. It pays the same attention to a Heartbleed sample as it does to a BENIGN sample, despite Heartbleed being approximately 10,000 times rarer. This causes the model to remain dominated by majority class patterns, leading to poor detection of rare attacks.

The IAA-Transformer (Imbalance-Aware Attention Transformer) addresses class imbalance through two complementary mechanisms.

First, weighted cross entropy loss is used during training instead of standard cross entropy. Each class is assigned a weight inversely proportional to its frequency in the original dataset. Rare classes receive higher weights, making mistakes on those classes more expensive in the loss function.

Second, a class frequency bias is added to the output logits after the model produces its 15 class scores. The bias term is formulated as alpha * log(1/f_c) where f_c is the frequency of class c in the original training data and alpha is a learnable scaling parameter constrained to be positive. When alpha = 0, the model reduces to a standard Transformer, serving as the ablation study baseline.

## Dataset

CICIDS-2017 (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset)
- 2,830,743 network flow records
- 79 features (reduced to 71 after preprocessing)
- 15 attack classes
- Severe class imbalance: BENIGN = 80.3%, Heartbleed = 11 samples

## Models Compared

1. Random Forest
2. XGBoost
3. 1D-CNN
4. BiLSTM
5. Standard Transformer
6. IAA-Transformer (proposed contribution)

## Key Results

IAA-Transformer consistently improves XSS recall across all experiments, averaging 0.71 across 5-fold cross validation compared to 0.58 for the standard Transformer baseline.

## Setup

```bash
conda create -n iads python=3.11
conda activate iads
pip install torch torchvision numpy pandas scikit-learn imbalanced-learn xgboost jupyterlab
```

## Project Structure

```
iads-project/
  data/
    processed/       <- cleaned train/val/test splits
  notebooks/         <- all experiment notebooks (01-14)
  models/            <- saved model weights
  results/           <- classification reports and figures
  scripts/           <- Narval HPC training scripts
```

## Author

Kritish Dhungel
MSc Computational Sciences, Laurentian University
Supervisor: Dr. Kalpdrum Passi
```
