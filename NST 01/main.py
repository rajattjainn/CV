import os

import torchvision.models as models
import torchvision.transforms as transforms
import torch as torch
from torchinfo import summary

from PIL import Image

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
STYLE_DIR = os.path.join(CURR_DIR, "Data", "style-images")
CONTENT_DIR = os.path.join(CURR_DIR, "Data", "content-images")

IMAGE_TRANSFORM = transforms.Compose(
    [transforms.Resize([100,100]),
    transforms.ToTensor()]
    )

vgg19 = models.vgg19(pretrained=True)


CONTENT_FILE = os.path.join(CONTENT_DIR, "taj_mahal.jpg")
STYLE_FILE = os.path.join(STYLE_DIR, "vg_starry_night.jpg")

CONTENT_IMAGE = Image.open(CONTENT_FILE)
STYLE_IMAGE = Image.open(STYLE_FILE)

CONTENT_TENSOR = IMAGE_TRANSFORM(CONTENT_IMAGE)
STYLE_TENSOR = IMAGE_TRANSFORM(STYLE_IMAGE)

print (CONTENT_TENSOR.size())
print (STYLE_TENSOR.size())

OUTPUT_TENSOR = torch.randn([3,100,100])
print (OUTPUT_TENSOR.size())


summary(vgg19)

