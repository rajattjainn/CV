from fileinput import filename
import os
import torchvision.utils as utils
import torchvision.transforms as transforms

from PIL import Image

def save_image_from_tensor(file_tensor, dir_path, output_file_name, file_format):
    if not os.path.isdir:
        os.makedirs(dir_path)
    utils.save_image(file_tensor, os.path.join(dir_path, output_file_name), format = file_format)
    return os.path.join(dir_path, output_file_name)

def get_input_image_transform(image_pixel):
    #ToDO: accomodate for rectangular images, adjust height according to width
    image_transform = transforms.Compose(
        [transforms.Resize([image_pixel,image_pixel]),
        transforms.ToTensor()]
        )

    return image_transform

def convert_image_to_tensor(dir_path, file_name):
    input_image = Image.open(os.path.join(dir_path, file_name))
    input_tensor = get_input_image_transform()(input_image)
    return input_tensor

