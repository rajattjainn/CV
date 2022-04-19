import os
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

import datasets

LAYER_TYPE = "layer_type"

def get_image_transform():
    image_transform = transforms.Compose([
            transforms.Resize([416, 416]),
            transforms.ToTensor()])

    return image_transform

def image_to_tensor(image_path):
    """
    Converts the input image to a tensor. Before the output is returned, the tensor is 
    divided by 255 as a file contains values from 0 to 255, but the operations are 
    performed on values from 0 to 1.
    
    """
    image = Image.open(image_path)
    tform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((416, 416))])
    img_tensor = tform(image)
    img_tensor = img_tensor/255
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def get_dataloader(image_folder, label_folder, shuffle = False):
    image_transform = get_image_transform()
    train_data = datasets.ObjectDataSet(image_folder, label_folder_path = label_folder, transform=image_transform)
    train_dataloader = DataLoader(train_data, batch_size = 8, shuffle = shuffle)
    return train_dataloader

def read_classes(classes_file):
    """
    Parses the config file. 
    @param cfg_file: the file containing configuration for the neural net
    @output: a list of dictionaries, each dictionary corresponding to one "module" in the cfg file
    """
    with open (classes_file) as file:
        lines = [line.lstrip().rstrip() for line in file.readlines()]
    return lines

def draw_rectangle(image_path, detections, classes):
    file_name = os.path.basename(image_path)
    source_img = Image.open(image_path).convert("RGB")
    width, height = source_img.size
    
    # Find the ratio between original width/height and width/height of the input image
    x_scale = width/416
    y_scale = height/416
    draw = ImageDraw.Draw(source_img)
    for detection in detections:
        # randomly pick a BB color
        rand_color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]

        draw.rectangle(((detection[0].item() * x_scale, detection[1].item() * y_scale), 
            (detection[2].item() * x_scale, detection[3].item() * y_scale)), outline = rand_color, fill=None)

        draw.text((detection[0].item(), detection[1].item()), classes[int(detection[6].item())] + 
            ", Confidence: " + "{0:.2f}".format((detection[5].item())), fill = rand_color)
    
    if not os.path.exists('det'):
        os.makedirs('det')
    det_path = os.path.join("det", file_name)
    source_img.save(det_path, "JPEG")

def parse_cfg(cfg_file):
    """
    Parses the config file. 
    @param cfg_file: the file containing configuration for the neural net
    @output: a list of dictionaries, each dictionary corresponding to one "module" in the cfg file
    """
    with open (cfg_file) as file:
        lines = [line.lstrip().rstrip() for line in file.readlines()]
    
    layer_dic_list = []
    layer_dic = {}
    for line in lines:
        if len(line) != 0 and not line.startswith("#"):
            if line.startswith("["):
                if len(layer_dic) != 0:
                    layer_dic_list.append(layer_dic)
                    layer_dic = {}
                layer_dic[LAYER_TYPE] = line[1:-1]
            else:
                key_value_pair = line.split("=")
                layer_dic[key_value_pair[0].rstrip()] = key_value_pair[1].lstrip()
    
    layer_dic_list.append(layer_dic)

    return layer_dic_list

def get_anchors(anchor_string, mask):
    """
    @param anchor_string: a list of strings, every 2 items denoting an anchor 
    @param mask: once anchor_string has been converted into a list of tuples, with 2 elemets in each tuple, mask defines the indexes of items in the above list that have to be used for anchors   
    @return anchor_list: a list of int tuples, each tuple defining the anchor

    """
    #TODO: Rewrite this function
    anc_list = []
    i = 0

    while i < len(anchor_string) - 1:
        anch = (int(anchor_string[i]), int(anchor_string[i+1]))
        anc_list.append(anch)
        i = i + 2
    anchor_list = []

    for item in range (len(mask)):
        anchor_list.append(anc_list[int(mask[item])])

    return anchor_list

def get_mesh_grid(grid_size):
    """
    Returns 2 tensors which can be added to 0th and 1st column of the img tensor
    """
    x_range = torch.arange(0, grid_size)
    y_range = torch.arange(0, grid_size)

    # intentionally y,x. in our operation, we'll traverse across the x axis first and then across y-axis.
    y,x = torch.meshgrid(x_range, y_range)

    # repeat the thrice because each yolo output cell contains data for 3 bounding boxes. 
    # 0th and 1st column in the img tensor represents the cx and cy values In order to add
    # the offsets, the tensors need to converted to column tensors after repeat
    x_cord_tensor = x.contiguous().view(-1,1).repeat(1,3).view(-1,1)
    y_cord_tensor = y.contiguous().view(-1,1).repeat(1,3).view(-1,1)
    return x_cord_tensor, y_cord_tensor