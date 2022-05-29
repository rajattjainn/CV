from torch import nn as nn

# Following DCGANGenerator is a fork from the Pytorch tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class DCGANGenerator(nn.Module):
    def __init__(self, ngpu, latent_vector_size, ftr_map_size_gn, op_chnls, bias=False) -> None:
        super(DCGANGenerator).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_vector_size, ftr_map_size_gn * 8, 4, 1, 0, bias = bias),
            nn.BatchNorm2d(ftr_map_size_gn*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ftr_map_size_gn * 8, ftr_map_size_gn * 4, 4, 2, 1, bias = bias),
            nn.BatchNorm2d(ftr_map_size_gn * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ftr_map_size_gn * 4, ftr_map_size_gn * 2, 4, 2, 1, bias = bias),
            nn.BatchNorm2d(ftr_map_size_gn * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ftr_map_size_gn * 2, ftr_map_size_gn, 4, 2, 1, bias = bias),
            nn.BatchNorm2d(ftr_map_size_gn),
            nn.ReLU(True),

            nn.ConvTranspose2d(ftr_map_size_gn, op_chnls, 4, 2, 1, bias = bias),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)