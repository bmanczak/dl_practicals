#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_shared_course
#SBATCH --job-name=example

cd /home/lgpu0268/1_mlp_cnn/code
python train_convnet_pytorch.py 
