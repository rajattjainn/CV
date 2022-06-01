import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

import discriminator as dsc
import generator as gnr
import data_utils

ngpu = 1
op_chnls = 3
ftr_map_size_dc = 64
ftr_map_size_gn = 64
latent_vector_size = 100
lr = 0.001
beta1 = 0.9
beta2 = 0.999
num_epochs = 5

data_dir = "data/celeba"
image_size = 64
batch_size = 128
num_workers = 0

netD = dsc.DCGANDiscriminator(ngpu, op_chnls, ftr_map_size_dc)
netG = gnr.DCGANGenerator(ngpu, latent_vector_size, ftr_map_size_gn, op_chnls)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = lr, betas = (beta1, beta2))
optimizerG = optim.Adam(netG.parameters(), lr = lr, betas = (beta1, beta2))

fixed_noise = torch.randn(64, latent_vector_size, 1, 1)

real_label_identifier = 1
fake_label_identifier = 0

dataloader = data_utils.get_datloader(data_dir, image_size, batch_size, True, num_workers)
G_losses = []
D_losses = []
gen_image_list = []
total_iters = num_epochs * (len(dataloader) * batch_size)

for epoch in range (num_epochs):
    for i, data in enumerate(dataloader, 0):
        # get rid of any residual gradients
        netD.zero_grad()
        netG.zero_grad()

        # getitem in ImageFolder returns 2 objects - image tensor and labels. 
        # Retrieve image tensor
        imgs = data[0]
        batch_size = len(imgs)
        
        # create real_label and fake_label tensors to be used for calculating loss 
        real_label = torch.full((batch_size, 1), real_label_identifier, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), fake_label_identifier, dtype=torch.float32)


        ##################################
        ##    Training Discriminator    ##
        ################################## 
        
        # Train discriminator on real data
        output = netD(imgs).view(-1).unsqueeze(1)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        # Train discriminator on fake data
        # Generate fake data first
        noise = torch.randn(batch_size, latent_vector_size, 1, 1, dtype=torch.float32)
        fake_imgs = netG(noise)
        output = netD(fake_imgs.detach()).view(-1).unsqueeze(1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()

        # get total loss for discriminator
        errD = errD_real + errD_fake
        lossD = errD.item()

        # step through the optimizer for discriminator
        optimizerD.step()


        ##################################
        ##      Training Generator      ##
        ################################## 
        
        # as we have applied optimizer.step on discriminator once,
        # we need to generate the output from discriminator once again.
        output = netD(fake_imgs).view(-1).unsqueeze(1)
        
        # While training the generator, real_label is the target
        errG = criterion(output, real_label)
        lossG = errG.item()
        errG.backward()
        
        # step through the optimizer for generator
        optimizerG.step()
        
        if (i % 50 == 0):
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            print ("Epoch {}/{}, Iteration: {}/{} \tLoss_G: {lg:.4f} \tLoss_D: {ld:.4f}".
                    format(epoch+1, num_epochs, i+1, len(dataloader), lg=errG.item(), ld=errD.item()))

        if (i % 500 == 0):
            gen_image = netG(fixed_noise)
            gen_image_list.append(gen_image)
        print ("\n")

    netD_checkpoint = "checkpoint_netD_" + str(epoch) + ".pt"
    netG_checkpoint = "checkpoint_netG_" + str(epoch) + ".pt"
    torch.save(netD.state_dict(), netD_checkpoint)
    torch.save(netG.state_dict(), netG_checkpoint)


torch.save(netD.state_dict(), "netD_final.pt")
torch.save(netG.state_dict(), "netG_final.pt")

# plot the losses over iterations
iter_step = (int) (total_iters/len(G_losses))
iters = np.array(range(0, total_iters, iter_step))

f = plt.figure()
f.set_figwidth(8)
f.set_figheight(8)
plt.plot(iters, G_losses, color="red", label="G_loss")
plt.plot(iters, D_losses, color="yellow", label="D_loss")
plt.title("Losses vs Iterations")
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.legend()
plt.savefig("loss_iter.png")

# save generated images after 500 iterations to disk
gen_image_tensor = torch.cat(gen_image_list, 0)
imgs_in_row = (gen_image_list[0].size(0))
grid = make_grid(gen_image_tensor, nrow=imgs_in_row, padding = 5)
print (gen_image_tensor.size())
print (grid.size())
f = plt.figure(clear=True)
plt.imshow(grid.permute(1,2,0))
plt.axis("off")
plt.savefig("generated_images.png")

# save last batch of generated images to disk
f = plt.figure(clear=True)
grid = make_grid(gen_image_list[len(gen_image_list)-1], padding = 5)
plt.imshow(grid.permute(1,2,0))
plt.axis("off")
plt.savefig("last_generated.png")