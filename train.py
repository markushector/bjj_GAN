import torch
from model.models import Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn as nn
from tqdm.auto import tqdm


criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002

beta_1 = 0.5
beta_2 = 0.999
device = 'cpu'

print("Loading data in dataloader. ")

train_transforms = transforms.Compose([
    transforms.Resize((28, 28)), #224, 224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

image_data = ImageFolder('../Bjj_Project/data/train_data',transform=train_transforms)
dataloader = DataLoader(image_data, batch_size=batch_size, shuffle=True)

print("Setting up generator and discriminator networks. ")

gen = Generator(input_dim=z_dim).to(device)
disc = Discriminator().to(device)

print("Setting up optimizers. ")

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

print("Initializing weights. ")

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

print("Starting training process. ")

n_epochs = 50
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(n_epochs):

    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        disc_opt.zero_grad()
        fake_noise = gen.get_noise(cur_batch_size, z_dim, device=device)
        fake = gen(fake_noise)

        disc_fake_pred = disc.forward(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc.forward(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()
        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step
        
        ## Update generator ##
        gen_opt.zero_grad()
        fake_noise_2 = gen.get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen.forward(fake_noise_2)
        disc_fake_pred = disc.forward(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator lossmean_generator_loss += gen_loss.item() / display_step
        if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            #show_tensor_images(fake)
            #show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

torch.save(gen.state_dict(), "generator.pt") #Saving model
