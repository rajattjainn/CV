
import torch.nn as nn

def parse_cfg(cfgfile):
    net_cfg = open(cfgfile, "rt")
    cfg_lines = list(net_cfg)
    cfg_lines = [line.strip() for line in cfg_lines]
    cfg_lines = list(filter(None, cfg_lines))
    block = {}
    blocks = []

    for line in cfg_lines:
        
        if line.startswith("#"):
            continue
        elif line.startswith("["):
            if len(block) > 0:
                blocks.append(block)
                block = {}
            section = line[1:-1]
            block["type"] = section.lstrip().rstrip()
        
        else:
            key, value = line.split("=")
            block[key.rstrip().lstrip()] = value.lstrip().rstrip()
        
    blocks.append(block)
    return blocks
            

def create_sequential_objects(block_list):
    prev_filter = 3
    filter_list = []
    filter_list.append(prev_filter)
    module_list = []

    for i, block in enumerate(block_list[1:]):
        module = nn.Sequential()

        block_type = block["type"]
        if block_type == "convolutional":

            if "batch_normalize" in (list(block.keys())):
                batch_normalize = block["batch_normalize"]
                bias = False
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
            module.add_module("conv_{0}".format(i), conv_block)
            prev_filter = filters
            filter_list.append(prev_filter)

            if batch_normalize:
                normalize_block = nn.BatchNorm2d(filters)
                module.add_module("batch_{0}".format(i), normalize_block)

            #ToDo: check this section once
            activation = block["activation"]
            if activation == "leaky":
                activaiton_block = nn.LeakyReLU()
                module.add_module("leakyrelu_{0}".format(i), activaiton_block)                
            elif activation == "linear":
                activaiton_block = nn.Linear(prev_filter, prev_filter)
                module.add_module("linear_{0}".format(i), activaiton_block)

            module_list.append(module)

        elif block_type == "shortcut":
            activation = block["activation"]

            if activation == "linear":
                activaiton_block = nn.Linear(prev_filter, prev_filter)
                module.add_module("linear_{0}".format(i), activaiton_block)
                filter_list.append(prev_filter)

            module_list.append(module)

    return module_list

            


blocks = parse_cfg("cfg/yolov3.cfg")
module_list = create_sequential_objects(blocks)
print ("---------checking starts----------")
for i, module in enumerate(module_list):
    print ("Index: " + str(i))
    print (module)
    print ("\n")