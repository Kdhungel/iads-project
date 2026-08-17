#!/bin/bash
#SBATCH --job-name=ca_leaky_sigmoid_123
#SBATCH --output=/scratch/kdhungel/iads-project/logs/cnnact_leaky_sigmoid_s123_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/cnnact_leaky_sigmoid_s123_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_cnn_activation.py leaky_sigmoid 123
