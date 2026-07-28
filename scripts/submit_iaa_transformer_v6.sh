#!/bin/bash
#SBATCH --job-name=iaa_v6
#SBATCH --output=/scratch/kdhungel/iads-project/logs/iaa_v6_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/iaa_v6_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_iaa_transformer_v6.py
