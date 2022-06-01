from torch import nn as nn

# Following DCGANDiscriminator is a fork from the Pytorch tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class DCGANDiscriminator(nn.Module):
    """
    A Discriminator class for the type DCGAN.
    """
    def __init__(self, op_chnls, ftr_map_size_dc, bias = False):
        """
        init function to create a DCGAN discriminator object.

        This function initializes a Sequential which is used during the forward function.

        Keyword Arguments:
        op_chnls: number of input image channels
        ftr_map_size_dc: depth of the discriminator input feature map
        bias: bias value for batch norm
        """
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(op_chnls, ftr_map_size_dc, 4, 2, 1, bias=bias),
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
        """
        Forward function of the DCGAN Discriminator object

        Keyword Arguments:
        input: the input image which has to identified as original or fake

        Returns:
        A binary value whether the image is real or fake.
        """
        return self.main(input)