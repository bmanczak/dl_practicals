import torch 
from train_pl import GAN
import os
import torchvision

os.chdir("/home/lgpu0268/DlCourse2020/3_generative/part2")
PATH = "/home/lgpu0268/DlCourse2020/3_generative/part2/GAN_logs/lightning_logs/version_7145603/checkpoints/epoch=249.ckpt"
model = GAN.load_from_checkpoint(PATH)

interpolation_steps = 5
batch_size = 4 

sample_imgs = model.interpolate(batch_size, interpolation_steps)
print("Shape of interpolation sample", sample_imgs.shape)
grid1 = torchvision.utils.make_grid(sample_imgs, nrow=interpolation_steps+2, normalize = True, range = (-1,1))

torchvision.utils.save_image(grid1, fp = "interpolation.jpg", nrow=interpolation_steps+2)

print(os.getcwd())