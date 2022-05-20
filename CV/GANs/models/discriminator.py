from torch import nn as nn


class DCGANDiscriminator(nn.Module):
    def __init__(self, ngpu, img_chnls, ftr_map_size_dc, bias = False):
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(img_chnls, ftr_map_size_dc, 4, 2, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ftr_map_size_dc, ftr_map_size_dc * 2, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ftr_map_size_dc * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ftr_map_size_dc * 2, ftr_map_size_dc * 4, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ftr_map_size_dc * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ftr_map_size_dc * 4, ftr_map_size_dc * 8, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ftr_map_size_dc * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ftr_map_size_dc * 8, 1, 4, 1, 0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)