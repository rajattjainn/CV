from locale import normalize
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import discriminator as dc
import generator as gn
import data_utils
import common_utils
import loss_utils

dataroot = "data/celeba"
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Number of channels in the training images. For color images this is 3
op_chnls = 3 
ftr_map_size_dc = 64
ftr_map_size_gn = 64
latent_vector_size = 100
# Number of workers for dataloader
workers = 2
workers = 0
# Batch size during training
batch_size = 128
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64


dataloader = data_utils.get_datloader(dataroot, image_size, batch_size, True, workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >0) else "cpu")

real_batch = next(iter(dataloader))
print (type(real_batch))
print (len(real_batch))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], 
        padding=2,normalize=True).cpu(), (1,2,0)))
# plt.show()


netD = dc.DCGANDiscriminator(ngpu, op_chnls, ftr_map_size_dc).to(device)
netG = gn.DCGANGenerator(ngpu, latent_vector_size, ftr_map_size_gn, op_chnls).to(device)

if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

netG.apply(common_utils.weights_init)
netD.apply(common_utils.weights_init)

criterion = loss_utils.get_bce_loss()

fixed_noise = torch.randn(64, latent_vector_size, 1, 1, device = device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, 0.999))

 
img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        real_label = torch.full((b_size,), real_label, dtype = torch.float, device = device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, real_label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, latent_vector_size, 1, 1, device=device)
        fake = netG(noise)
        fake_label = torch.full((b_size,), fake_label, dtype = torch.float, device = device)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()


        netG.zero_grad()
        output = netD(fake).view(-1)
        errG = criterion(output, real_label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()


        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 ==0) or ((epoch == num_epochs -1) and (i == len(dataloader) -1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding = 2, normalize=True))
        
        iters += 1

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()