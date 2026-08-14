#!/bin/bash
#SBATCH --job-name=sweep_d0.0
#SBATCH --output=/scratch/kdhungel/iads-project/logs/sweep_d0.0_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/sweep_d0.0_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_dropout_sweep.py 42 0.0
