#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --output=/scratch/kdhungel/iads-project/logs/exp2_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/exp2_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_experiments.py exp2 42
