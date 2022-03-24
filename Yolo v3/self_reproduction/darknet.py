
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

        elif block_type == "shortcut":
            # An alternative to EmptyLayer is to add nothing and handle this condition in the forward pass
            # by applying concatenation. However, EmptyLayer is required to make sure the length of 
            # block_list and module_list is same and the indexing is ordered. This ordering helps in
            #  the forward function of Darknet neural network. 
            empty_block = EmptyLayer()
            module_list.append(module)
            module.add_module(block_type + "_{0}".format(index), empty_block)

        elif block_type == "route":
            # An alternative to EmptyLayer is to add nothing and handle this condition in the forward pass
            # by applying concatenation. However, EmptyLayer is required to make sure the length of 
            # block_list and module_list is same and the indexing is ordered. This ordering helps in
            #  the forward function of Darknet neural network. 
            empty_block = EmptyLayer()
            module_list.append(module)
            module.add_module(block_type + "_{0}".format(index), empty_block)

            layers = block["layers"]    
            if "," in layers:    
                layers = [int(x) for x in layers.split(",")] 
                filters = filter_list[index + int(layers[0])] + filter_list[int(layers[1])]
            else:
                filters = filter_list[index + int(layers)]
    
        elif block_type == 'yolo':
            # mask tells which anchor boxes to use. Yolo detects unage at 3 levels, each level has 3 anchor boxes.
            mask = [int(x) for x in block["mask"].split(",")] 
            anchor_iter = iter([int (x) for x in block["anchors"].split(",")])
            anchor_tuples = [*zip(anchor_iter, anchor_iter)]
            anchors = [anchor_tuples[x] for x in mask]
            
            detection_block = DetectionLayer(anchors)
            module.add_module("yolo_{0}".format(index), detection_block)
            module_list.append(module)

        prev_filter = filters
        filter_list.append(prev_filter)
    
    return module_list

def get_test_input(input_image):
    img = cv2.imread(input_image)
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = torch.Tensor(img_)                     # Convert to Variable
    return img_

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
        block_list = self.block_list[1:]
        write = 0
        for index, block in enumerate(self.module_list):
            #ToDo: find a cleaner way to do this action
            print ("\n\nindex: " + str(index))
            block_type = block_list[index]["type"]

            if block_type == "convolutional" or block_type == "upsample":
                output = block(prev_output)
            
            elif block_type == "shortcut":
                relative_index = int(block_list[index]["from"])
                absolute_index = index + relative_index
                output = output_ftr_map_list[absolute_index]

                
            elif block_type == "route":
                layers = block_list[index]["layers"]
                
                if "," in layers:    
                    layers = [int(x) for x in layers.split(",")]
                    output_1 = output_ftr_map_list[index + int(layers[0])]
                    output_2 = output_ftr_map_list[int(layers[1])]
                    # print ("2 layers, absolute indexes: " + str(index + int(layers[0])) + ", " + str(int(layers[1])))    
                    output = torch.cat((output_1, output_2), 1)
                else:
                    output = output_ftr_map_list[index + int(layers)]
                    # print ("single layer, absolute index: " + str(index + int(layers)))    
                  
            
            elif block_type == "yolo":
                mask = [int(x) for x in self.block_list[index + 1]["mask"].split(",")]
                anchors = iter([int(x) for x in self.block_list[index + 1]["anchors"].split(",")])
                anchors = [*zip(anchors, anchors)]
                anchors = [anchors[index] for index in mask]
                num_classes = int(self.block_list[index + 1]["classes"])
                height = int(net_info["height"])
                output = utils.transform_prediction(prev_output, anchors, num_classes, height)
            

                if write:
                    detection_tensor = torch.cat((detection_tensor, output), 1)
                else:
                    detection_tensor = output
                    write = 1
            
            prev_output = output
            output_ftr_map_list.append(output)

        return detection_tensor    

    def load_weights(self, weightfile):
        #Open the weights file
        fp = open(weightfile, "rb")
    
        #The first 5 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]   
        
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
    
            #If module_type is convolutional load weights
            #Otherwise ignore.
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
            
                conv = model[0]
                
                
                if (batch_normalize):
                    bn = model[1]
        
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
        
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
        
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
        
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
        
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


# blocks = parse_cfg("cfg/yolov3.cfg")

# module_list = create_modules(blocks)
# print ("---------checking starts----------")
# for i, module in enumerate(module_list):
#     print ("Index: " + str(i))
#     print (module)
#     print ("\n")


# create_modules(parse_cfg("cfg/yolov3.cfg"))
net = Darknet("cfg/yolov3.cfg")
net(get_test_input("imgs/dog.jpg"))