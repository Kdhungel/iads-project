#!/bin/bash
#SBATCH --job-name=ba_relu_sigmoid_456
#SBATCH --output=/scratch/kdhungel/iads-project/logs/binact_relu_sigmoid_s456_%j.out
#SBATCH --error=/scratch/kdhungel/iads-project/logs/binact_relu_sigmoid_s456_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --account=def-kpassi

source ~/iads-env/bin/activate
python /scratch/kdhungel/iads-project/scripts/train_binary_activation.py relu_sigmoid 456
