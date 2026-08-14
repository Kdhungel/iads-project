#!/bin/bash
#SBATCH --job-name=tier1
#SBATCH --output=/scratch/kdhungel/iads-project/logs/tier1_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/tier1_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_tier1_classical.py
