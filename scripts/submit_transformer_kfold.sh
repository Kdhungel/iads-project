#!/bin/bash
#SBATCH --job-name=transformer_kfold
#SBATCH --output=/scratch/kdhungel/iads-project/logs/transformer_kfold_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/transformer_kfold_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_transformer_kfold.py
