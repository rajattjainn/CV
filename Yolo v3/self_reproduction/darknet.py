
import numpy as np
import torch.nn as nn
import torch as torch
import cv2

import utils

class EmptyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors) -> None:
        super().__init__()
        self.anchors = anchors

def parse_cfg(cfgfile):
    """
    Reads the configuration file that contains the architecture of the algorithm.

    Creates a dictionary object for each block with various attributes working as the key(s).
    It should be noted that a block is not equal to a single layer like convolutional or linear.
    For example a convolutional block can have normalize, convolutional and linear layers. 

    Adds each dictionary object to a list and returns that list.

    @param cfgfile: the configuration file
    @return: a list of dictionaries, each dictionary depicting a neural (sequential) block.
    """
    net_cfg = open(cfgfile, "rt")
    cfg_lines = list(net_cfg)
    cfg_lines = [line.strip() for line in cfg_lines]
    cfg_lines = list(filter(None, cfg_lines))
    block = {}
    blocks = []

    for line in cfg_lines:
        
        if line.startswith("#"): #comments in the cfg file
            continue
        elif line.startswith("["):
            if len(block) > 0:
                blocks.append(block)
                block = {}
            type = line[1:-1]
            block["type"] = type.lstrip().rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip().lstrip()] = value.lstrip().rstrip()
    blocks.append(block)

    return blocks

def create_modules(block_list):
    """ 
    Create building blocks of the neural network. 
    @param block_list: a list of dictionaries, each dictionary depicting a neural (sequential) block.
    @output module_list: list of modules, each module an object of nn.Sequential class.
    """
    prev_filter = 3
    filter_list = []
    filter_list.append(prev_filter)
    module_list = []

    for index, block in enumerate(block_list[1:]):
        module = nn.Sequential()

        block_type = block["type"]
        if block_type == "convolutional":
            
            if "batch_normalize" in (list(block.keys())):
                batch_normalize = block["batch_normalize"]
                bias = False # Batchnormalization already includes the addition of the bias term. Refer https://stackoverflow.com/a/46256860/1937725
            else:
                batch_normalize = 0
                bias = True
            
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            pad = int(block["pad"])
            if pad:
                pad = kernel_size //2 # as explained by the author of cfg file at (https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers)
            conv_block = nn.Conv2d(prev_filter, filters, kernel_size, stride = stride, padding = pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv_block)
            prev_filter = filters
            filter_list.append(prev_filter)

            if batch_normalize:
                normalize_block = nn.BatchNorm2d(filters)
                module.add_module("batch_{0}".format(index), normalize_block)

            #ToDo: check this section once
            activation = block["activation"]
            if activation == "leaky":
                activaiton_block = nn.LeakyReLU()
                module.add_module("leakyrelu_{0}".format(index), activaiton_block)
            # adding this layer raises an exception. multiple implementations of yolov3 haven't implemented 
            # linear activation in convolutional layer. Researching in parallel why this hasn't been done.

            # elif activation == "linear":
            #     activaiton_block = nn.Linear(prev_filter, prev_filter)
            #     module.add_module("linear_{0}".format(index), activaiton_block)

            module_list.append(module)

        elif block_type == "upsample":
            stride = int(block["stride"])
            upsample_block = nn.Upsample(scale_factor = stride)
            module.add_module("upsample_{0}".format(index), upsample_block)

            module_list.append(module)

        elif block_type == "shortcut" or block_type == "route":
            # An alternative to EmptyLayer is to add nothing and handle this condition in the forward pass
            # by applying concatenation. However, EmptyLayer is required to make sure the length of 
            # block_list and module_list is same and the indexing is ordered. This ordering helps in
            #  the forward function of Darknet neural network. 
            empty_block = EmptyLayer()
            module_list.append(module)
            module.add_module(block_type + "_{0}".format(index), empty_block)
        
        elif block_type == 'yolo':
            # mask tells which anchor boxes to use. Yolo detects unage at 3 levels, each level has 3 anchor boxes.
            mask = [int(x) for x in block["mask"].split(",")] 
            anchor_iter = iter([int (x) for x in block["anchors"].split(",")])
            anchor_tuples = [*zip(anchor_iter, anchor_iter)]
            anchors = [anchor_tuples[x] for x in mask]
            
            detection_block = DetectionLayer(anchors)
            module.add_module("yolo_{0}".format(index), detection_block)
            module_list.append(module)

    return module_list

class Darknet(nn.Module):
    def __init__(self, cfgfile) -> None:
        super().__init__()
        self.block_list = parse_cfg(cfgfile)
        self.module_list = create_modules(self.block_list)

    def forward(self, input_data):
        net_info = self.block_list[0]
        output_ftr_map_list = []
        # setting prev_output equal to input_data for 0th index.
        prev_output = input_data
        print ("modue list length")
        print (len(self.module_list[1:]))
        print ("\n\n module list -------------")
        print (self.module_list)

        write = 0
        for index, block in enumerate(self.module_list[1:]):
            #ToDo: find a cleaner way to do this action
            print ("\n\nindex: " + str(index))
                        
            block_type = self.block_list[index + 1]["type"]
            
            if block_type == "convolutional" or block_type == "upsample":
                print (block_type)
                print ("prev output size:" )
                print (prev_output.size())
                print (self.module_list[index])
                output = self.module_list[index](prev_output)
                print ("output size:")
                print (output.size())
            elif block_type == "shortcut":
                relative_index = int(self.block_list[index + 1]["from"])
                absolute_index = index + relative_index
                output = output_ftr_map_list[absolute_index]
                # print ("\n\nshortcut")
                # print ("relative_index: " + str(relative_index))
                # print ("absolute_index: " + str(absolute_index))
                
            elif block_type == "route":
                layers = self.block_list[index + 1]["layers"]
                # print ("\n\nroute")
                print ("layers")
                print (layers)
                print ("\n\n")
                
                if "," in layers:    
                    layers = [int(x) for x in layers.split(",")]
                    output_1 = output_ftr_map_list[index + int(layers[0])]
                    output_2 = output_ftr_map_list[int(layers[1])]
                    # print ("2 layers, absolute indexes: " + str(index + layers[0]) + ", " + str(layers[1]))    
                    output = torch.cat((output_1, output_2), 1)
                    # print ("output tensor: ")
                    # print (output)
                    # print (output.size())
                else:
                    # print ("single layer, absolute index: " + str(index + layers[0]))    
                    output = output_ftr_map_list[index + int(layers)]
                    # print ("output tensor: ")
                    # print (output)
                    # print (output.size())
            
            elif block_type == "yolo":
                print ("\n\nyolo")
                mask = [int(x) for x in self.block_list[index + 1]["mask"].split(",")]
                anchors = iter([int(x) for x in self.block_list[index + 1]["anchors"].split(",")])
                anchors = [*zip(anchors, anchors)]
                anchors = [anchors[index] for index in mask]
                num_classes = int(self.block_list[index + 1]["classes"])
                height = int(net_info["height"])
                print ("going in transform")
                output = utils.transform_prediction(prev_output, anchors, num_classes, height)


                if write:
                    detection_tensor = torch.cat((detection_tensor, output), 1)
                else:
                    detection_tensor = output
                    write = 1
            
            prev_output = output
            output_ftr_map_list.append(output)

        print ("\n\nending for loop. size of return tensor: ")
        print (detection_tensor.size())
        return detection_tensor    


def get_test_input(input_image):
    img = cv2.imread(input_image)
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = torch.Tensor(img_)                     # Convert to Variable
    return img_


# blocks = parse_cfg("cfg/yolov3.cfg")

# module_list = create_modules(blocks)
# print ("---------checking starts----------")
# for i, module in enumerate(module_list):
#     print ("Index: " + str(i))
#     print (module)
#     print ("\n")


net = Darknet("cfg/yolov3.cfg")
net(get_test_input("imgs/dog.jpg"))