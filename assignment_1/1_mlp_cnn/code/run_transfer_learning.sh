#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared_course
#SBATCH --time=1:00:00

cd /home/lgpu0268/1_mlp_cnn/code
python transfer_learning.py
