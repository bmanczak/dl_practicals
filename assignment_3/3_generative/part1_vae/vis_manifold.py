import torch
from train_pl import VAE
from utils import visualize_manifold
import torchvision
import os

os.chdir("/home/lgpu0268/DlCourse2020/3_generative/part1")
model = VAE.load_from_checkpoint("/home/lgpu0268/DlCourse2020/3_generative/part1/VAE_logs/lightning_logs/version_6/checkpoints/epoch=37.ckpt")

grid_size = 20

img_grid = visualize_manifold(model.decoder, grid_size)
torchvision.utils.save_image(img_grid, fp = "manifold.jpg", nrow=grid_size)
