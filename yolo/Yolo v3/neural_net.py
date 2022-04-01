from sysconfig import parse_config_h
import torch
from torch import nn as nn
import numpy as np

LAYER_TYPE = "layer_type"

class EmptyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

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

def create_module_list(layer_dic_list):
    # ModuleList is being used to enable transfer learning we might want to work on later.
    module_list = nn.ModuleList()
    net_info = layer_dic_list[0]
    layer_dic_list = layer_dic_list[1:]

    prev_filter = 3
    filter_list = []

    for index, layer in enumerate(layer_dic_list):
        module = nn.Sequential()
        if layer[LAYER_TYPE] == "convolutional":
            out_filters = int(layer["filters"])
            kernel = int(layer["size"])
            stride = int(layer["stride"])
            pad = int(layer["pad"])
            activation = layer["activation"]

            #using the value of padding as defined: https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers
            if pad:
                padding = kernel // 2
            else:
                padding = 0
            
            try:
                batch_normlize = layer["batch_normalize"]
                bias = False
            except:
                bias = True
                batch_normlize = 0
            
            conv_module = nn.Conv2d(prev_filter, out_filters, kernel, stride = stride, padding = padding, bias = bias)
            module.add_module("conv_{0}".format(index), conv_module)

            if batch_normlize:
                batch_norm_module = nn.BatchNorm2d(out_filters)
                module.add_module("batchnorm_{0}".format(index), batch_norm_module)
            
            if activation == "leaky":
                activation_module = nn.LeakyReLU()
                module.add_module("leaky_{0}".format(index), activation_module)

        if layer[LAYER_TYPE] == "shortcut":
            shortcut_module = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut_module)

        if layer[LAYER_TYPE] == "upsample":
            stride = layer["stride"]
            upsample_module = nn.Upsample(scale_factor = stride, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample_module)

        if layer[LAYER_TYPE] == "route":
            route_module = EmptyLayer()
            module.add_module("route_{0}".format(index), route_module)

        if layer[LAYER_TYPE] == "yolo":
            yolo_module = EmptyLayer()
            module.add_module("yolo_{0}".format(index), yolo_module)

        module_list.append(module)
        prev_filter = out_filters
        filter_list.append(prev_filter)

    
    return net_info, module_list

def get_mesh_grid(grid_size):
    #Rewrite this part
    x_range = np.arange(0, grid_size)
    y_range = np.arange(0, grid_size)
    x,y = np.meshgrid(x_range, y_range)
    x = x.repeat(3,1)
    y = y.repeat(3,1)
    x_cord_tensor = torch.tensor(x)
    y_cord_tensor = torch.tensor(y)
    x_cord_tensor = x_cord_tensor.view(-1,1)
    y_cord_tensor = y_cord_tensor.view(-1,1)

    return x_cord_tensor, y_cord_tensor

def transform_yolo_output(input, anchors, cnf_thres):
    batch_size = input.size(0)
    grid_size = input[0].size(1)

    #rewrite
    anc_tensor_exists = False
    for item in anchors:
        if anc_tensor_exists:
            anc_tensor = torch.cat((anc_tensor, torch.tensor(item)),0)
        else:
            anc_tensor = torch.tensor(item)
            anc_tensor_exists = True
            
    anc_tensor = anc_tensor.view(-1, 2).repeat(grid_size*grid_size,1)


    for i in range(batch_size):
        img = input[i]
        
        img = img.transpose(0,2).contiguous()
        img = img.view(grid_size * grid_size, -1)
        img = img.view(grid_size * grid_size * 3, -1)

        img[:, 0] = torch.sigmoid(img[:, 0])
        img[:, 1] = torch.sigmoid(img[:, 1])
        img[:, 4] = torch.sigmoid(img[:, 4])
        
        x_cord_tensor, y_cord_tensor = get_mesh_grid(grid_size)

        img[:, 0] = img[:, 0] + x_cord_tensor.squeeze(1)
        img[:, 1] = img[:, 1] + y_cord_tensor.squeeze(1)
        
        img[:,2] = anc_tensor[:,0] * torch.exp(img[:, 2])
        img[:,3] = anc_tensor[:,1] * torch.exp(img[:, 3])

        img = img[img[:, 4] > cnf_thres]

        max_values = torch.max(img[:,5:], 1)
        img = torch.cat((img[:, :5], max_values[0].unsqueeze(1), max_values[1].unsqueeze(1)), 1)

        print (img.size())


class Yolo3(nn.Module):
    def __init__(self, cfg_file):
        super().__init__()
        self.layer_dic_list = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_module_list(self.layer_dic_list)

    def forward(self, input):
        layer_dic_list = self.layer_dic_list[1:]
        module_list = self.module_list
        feature_map_list = []

        for index, layer_dic in enumerate(layer_dic_list):
            if layer_dic[LAYER_TYPE] == "convolutional":
                output = module_list[index](input)

            elif layer_dic[LAYER_TYPE] == "shortcut":
                from_layer = int(layer_dic["from"])
                abs_shrtct_layer = index + from_layer
                output = feature_map_list[abs_shrtct_layer]

            elif layer_dic[LAYER_TYPE] == "upsample":
                output = module_list[index](input)

            elif layer_dic[LAYER_TYPE] == "route":
                layers = layer_dic["layers"]
                
                if layers.contain(","):
                    layers = layers.split(",")
                    layer1 = int(layers[0])
                    layer2 = int(layers[1])

                    if layer1 < 0:
                        layer1 = index + layer1
                    if layer2 < 0:
                        layer2 = index + layer2

                    out1 = feature_map_list[layer1]
                    out2 = feature_map_list[layer2]
                    output = torch.cat((out1, out2), 1)
                else:
                    layer = int(layers)
                    absolute_route_layer = index + layer
                    output = feature_map_list[absolute_route_layer]

            elif layer_dic[LAYER_TYPE] == "yolo": 
                #rewrite from here
                anchors = layer_dic["anchors"].split(",")
                anc_list = []
                i = 0
                while i < len(anchors) - 1:
                    anch = (int(anchors[i]), int(anchors[i+1]))
                    anc_list.append(anch)
                    i = i + 2
                mask = layer_dic["mask"].split(",")
                anchors = []
                for item in range (len(mask)):
                    anchors.append(anc_list[int(mask[item])])

                #rewrite till here
                output = transform_yolo_output(input, anchors, cnf_thres = 0.5)
                
                break


            feature_map_list.append(output)
            input = output


input = torch.randn(1, 3, 416, 416)
net = Yolo3("assets/config.cfg")
net(input)