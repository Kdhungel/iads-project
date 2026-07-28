#!/bin/bash
#SBATCH --job-name=trans_no_dp
#SBATCH --output=/scratch/kdhungel/iads-project/logs/trans_no_dp_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/trans_no_dp_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_transformer_no_destport.py
