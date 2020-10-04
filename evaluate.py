import torch
from model.models import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = 'cpu'
batch = 25
z_dim = 64

gen = Generator(input_dim=z_dim).to(device=device)
gen.load_state_dict(torch.load("generator.pt"))

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):

    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

fake_noise = gen.get_noise(batch, z_dim, device=device)
generated_images = gen(fake_noise)

show_tensor_images(generated_images)
