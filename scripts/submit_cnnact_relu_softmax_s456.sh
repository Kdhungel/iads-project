#!/bin/bash
#SBATCH --job-name=ca_relu_softmax_456
#SBATCH --output=/scratch/kdhungel/iads-project/logs/cnnact_relu_softmax_s456_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/cnnact_relu_softmax_s456_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_cnn_activation.py relu_softmax 456
