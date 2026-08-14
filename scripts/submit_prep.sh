#!/bin/bash
#SBATCH --job-name=prep_exp
#SBATCH --output=/scratch/kdhungel/iads-project/logs/prep_exp_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/prep_exp_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/prep_experiments.py
