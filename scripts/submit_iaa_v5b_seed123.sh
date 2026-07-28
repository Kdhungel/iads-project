#!/bin/bash
#SBATCH --job-name=iaa_seed123
#SBATCH --output=/scratch/kdhungel/iads-project/logs/iaa_seed123_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/iaa_seed123_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_iaa_v5b_seed123.py
