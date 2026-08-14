#!/bin/bash
#SBATCH --job-name=exp4
#SBATCH --output=/scratch/kdhungel/iads-project/logs/exp4_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/exp4_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_exp4_classical.py
