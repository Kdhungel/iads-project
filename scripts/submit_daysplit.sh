#!/bin/bash
#SBATCH --job-name=daysplit
#SBATCH --output=/scratch/kdhungel/iads-project/logs/daysplit_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/daysplit_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_daysplit.py
