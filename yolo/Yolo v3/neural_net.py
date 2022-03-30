from sysconfig import parse_config_h
import torch
from torch import nn as nn

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
            print (line)
            if line.startswith("["):
                if len(layer_dic) != 0:
                    layer_dic_list.append(layer_dic)
                    layer_dic = {}
                layer_dic[LAYER_TYPE] = line[1:-1]
            else:
                key_value_pair = line.split("=")
                layer_dic[key_value_pair[0].rstrip()] = key_value_pair[1].lstrip()
    
    layer_dic_list.append(layer_dic)

def create_module_list(layer_dic_list):
    # ModuleList is being used to enable transfer learning we might want to work on later.
    module_list = nn.ModuleList()
    net_info = layer_dic_list[0]

    prev_filter = 3
    filter_list = []

    for index, layer in enumerate(layer_dic_list):
        if layer[LAYER_TYPE] == "convolutional":
            out_filters = layer["filters"]
            kernel = layer["size"]
            stride = layer["stride"]
            pad = layer["pad"]
            activation = layer["activation"]

            try:
                batch_normlize = layer["batch_normalize"]
                bias = False
            except:
                bias = True
                batch_normlize = 0
            
            conv_module = nn.Conv2d(prev_filter, out_filters, kernel, stride = stride, padding = pad, bias = bias)
            module_list.add_module("conv_{0}".format(index), conv_module)

            if batch_normlize:
                batch_norm_module = nn.BatchNorm2d(out_filters)
                module_list.add_module("batchnorm_{0}".format(index), batch_norm_module)
            
            if activation == "leaky":
                activation_module = nn.LeakyReLU()
                module_list.add_module("leaky_{0}".format(index, activation_module))

        if layer[LAYER_TYPE] == "shortcut":
            shortcut_module = EmptyLayer()
            module_list.add_module("shortcut_{0}".format(index), shortcut_module)

        if layer[LAYER_TYPE] == "upsample":
            stride = layer["stride"]
            upsample_module = nn.Upsample(scale_factor = stride, mode="bilinear")
            module_list.add_module("upsample_{0}".format(index, upsample_module))

        if layer[LAYER_TYPE] == "yolo":
            yolo_module = EmptyLayer()
            module_list.add_module("yolo_{0}".format(index), yolo_module)

        prev_filter = out_filters
        filter_list.append(prev_filter)

    
    return net_info, module_list


# class Yolo3(nn.Module):
#     def __init__(self, cfg_file) -> None:
#         super().__init__()
#         self.layer_dic_list = parse_config(cfg_file)
