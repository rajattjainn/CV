import os

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch as torch
import torch.nn as nn

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


# summary(vgg19)
print ("not printing vgg16 ")

NUM_ITERATIONS = 1000
MSE_LOSS = nn.functional.mse_loss
LEARNING_RATE = 1e-3
truncated_model = nn.Sequential(*list(vgg19.features)[:35])
summary(truncated_model)
print ("hola")
for iteration in range(NUM_ITERATIONS):
    content_features = truncated_model(CONTENT_TENSOR[None])
    output_features = truncated_model(OUTPUT_TENSOR[None])
    print ("iter : " + str(iteration) + "\n")

    OUTPUT_TENSOR = OUTPUT_TENSOR - LEARNING_RATE*MSE_LOSS(
        content_features, output_features)

utils.save_image(OUTPUT_TENSOR, os.path.join(CURR_DIR, "äbc.png"), format = "PNG")