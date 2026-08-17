#!/bin/bash
#SBATCH --job-name=ca_relu_sigmoid_42
#SBATCH --output=/scratch/kdhungel/iads-project/logs/cnnact_relu_sigmoid_s42_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/cnnact_relu_sigmoid_s42_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_cnn_activation.py relu_sigmoid 42
