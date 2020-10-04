import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, input_dim=10, im_chan=3, kernel_size=3, stride=2, hidden_dim=64):
        super(Generator, self).__init__()
        layer1 = nn.ConvTranspose2d(input_dim, hidden_dim*4)
        layer2 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2)
        layer3 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim)
        output_layer = nn.ConvTranspose2d(hidden_dim, im_chan)

    def forward(x):

        return



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
