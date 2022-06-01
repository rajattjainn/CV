import torch
import torch.nn as nn
import torch.optim as optim

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
num_epochs = 1

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
        print (real_label.size())
        print (fake_label.size())
        
        ##################################
        ##    Training Discriminator    ##
        # ################################ 
        
        # Train discriminator on real data
        output = netD(imgs).view(-1).unsqueeze(1)
        print (output.size())
        errD_real = criterion(output, real_label)
        errD_real.backward()

        # Train discriminator on fake data.
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
        # ################################ 
        # as we have applied optimizer.step on discriminator once,
        #  we need to generate the output from discriminator once again.
        output = netD(fake_imgs).view(-1).unsqueeze(1)
        # While training the generator, real_label is the target
        errG = criterion(output, real_label)
        lossG = errG.item()
        errG.backward()
        
        # step through the optimizer for generator
        optimizerG.step()
        
        
        print ("\n")
