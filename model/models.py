import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):


    def __init__(self, input_dim=10, im_chan=3, kernel_size=3, stride=2, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        """self.layer1 = nn.ConvTranspose2d(input_dim, hidden_dim*4, kernel_size=kernel_size, stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(hidden_dim*4)
        self.layer2 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(hidden_dim*2)
        self.layer3 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=kernel_size, stride=stride)
        self.batch_norm3 = nn.BatchNorm2d(hidden_dim)
        self.output_layer = nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=stride)"""

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(input_dim, hidden_dim*4, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=stride),
            nn.Tanh()

        )


    def forward(self, x):
        """x = nn.ReLU(self.batch_norm1(self.layer1(x)))
        x = nn.ReLU(self.batch_norm2(self.layer2(x)))
        x = nn.ReLU(self.batch_norm3(self.layer3(x)))
        x = nn.Tanh(self.output_layer(x))"""

        x = x.view(len(x), self.input_dim, 1, 1)
        x = self.gen(x)

        return x


    def get_noise(self, n_samples, z_dim, device='cpu'):

        return torch.randn(n_samples, z_dim, device=device)


class Discriminator(nn.Module):

    def __init__(self, im_chan=3, hidden_dim=16, kernel_size=4, stride=2):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Conv2d(im_chan, hidden_dim, kernel_size=kernel_size, stride=stride)
        self.batch_norm1 = nn.BatchNorm2d(hidden_dim)
        self.layer2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=kernel_size, stride=stride)
        self.batch_norm2 = nn.BatchNorm2d(hidden_dim*2)
        self.layer3 = nn.Conv2d(hidden_dim*2, 1, kernel_size=kernel_size, stride=stride)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = self.activation(self.batch_norm1(self.layer1(x)))
        x = self.activation(self.batch_norm2(self.layer2(x)))
        x = self.layer3(x)

        return x.view(len(x), -1)
