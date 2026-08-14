#!/bin/bash
#SBATCH --job-name=mp_s456
#SBATCH --output=/scratch/kdhungel/iads-project/logs/multipass_s456_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/multipass_s456_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_multipass.py 456
