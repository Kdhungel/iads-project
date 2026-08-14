#!/bin/bash
#SBATCH --job-name=verify_xgb
#SBATCH --output=/scratch/kdhungel/iads-project/logs/verify_xgb_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/verify_xgb_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/verify_xgb.py
