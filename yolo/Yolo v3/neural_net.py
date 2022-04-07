import numpy as np
import torch
from torch import nn as nn
import torchvision.ops as tvo
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import random
import os
import argparse

LAYER_TYPE = "layer_type"

class EmptyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

def read_classes(classes_file):
    """
    Parses the config file. 
    @param cfg_file: the file containing configuration for the neural net
    @output: a list of dictionaries, each dictionary corresponding to one "module" in the cfg file
    """
    with open (classes_file) as file:
        lines = [line.lstrip().rstrip() for line in file.readlines()]
    return lines

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
                batch_normalize = layer["batch_normalize"]
                bias = False
            except:
                bias = True
                batch_normalize = 0
            
            conv_module = nn.Conv2d(prev_filter, out_filters, kernel, stride = stride, padding = padding, bias = bias)
            module.add_module("conv_{0}".format(index), conv_module)

            if batch_normalize:
                batch_norm_module = nn.BatchNorm2d(out_filters)
                module.add_module("batchnorm_{0}".format(index), batch_norm_module)
            
            if activation == "leaky":
                activation_module = nn.LeakyReLU(0.1)
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
            prev_layers = layer["layers"]
            if "," in prev_layers:
                prev_layers = prev_layers.split(",")
                prev_layer1 = int(prev_layers[0])
                prev_layer2 = int(prev_layers[1])

                if prev_layer1 < 0:
                    prev_layer1 = index + prev_layer1
                if prev_layer2 < 0:
                    prev_layer2 = index + prev_layer2

                filter1 = filter_list[prev_layer1]
                filter2 = filter_list[prev_layer2]
                out_filters = filter1 + filter2
            else:
                prev_layers = int(prev_layers)
                out_filters = filter_list[prev_layers]

        if layer[LAYER_TYPE] == "yolo":
            yolo_module = EmptyLayer()
            module.add_module("yolo_{0}".format(index), yolo_module)

        module_list.append(module)
        prev_filter = out_filters
        filter_list.append(prev_filter)

    
    return net_info, module_list

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

def transform_yolo_output(input, anchors, height):
    input = input.float()
    grid_size = input[0].size(1)
    stride = height // input[0].size(2)
    
    anc_tensor = torch.tensor(anchors)
    anc_tensor = anc_tensor.repeat(grid_size*grid_size,1)
    anc_tensor = anc_tensor/stride

    img = input[0]
    
    img = img.view(-1, grid_size * grid_size)
    img = img.transpose(0,1).contiguous()
    img = img.view(grid_size * grid_size * 3, -1)
    

    # perform yolo calculations
    img[:, 0] = torch.sigmoid(img[:, 0])
    img[:, 1] = torch.sigmoid(img[:, 1])
    img[:, 4] = torch.sigmoid(img[:, 4])
    img[:, 5:] = torch.sigmoid(img[:, 5:]) 
    img[:,2] = anc_tensor[:,0] * torch.exp(img[:, 2])
    img[:,3] = anc_tensor[:,1] * torch.exp(img[:, 3])

    x_cord_tensor, y_cord_tensor = get_mesh_grid(grid_size)
    img[:, 0] = img[:, 0] + x_cord_tensor.squeeze(1)
    img[:, 1] = img[:, 1] + y_cord_tensor.squeeze(1)
        
    # multiply the coordinates by stride 
    img[:, :4] = img[:, :4] * stride
    
    return img.unsqueeze(0)

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

def image_to_tensor(image_path):
    image = Image.open(image_path)
    tform = transforms.Compose([transforms.PILToTensor(), transforms.Resize((416, 416))])
    img_tensor = tform(image)
    img_tensor = img_tensor/255
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

class Yolo3(nn.Module):
    def __init__(self, cfg_file):
        super().__init__()
        self.layer_dic_list = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_module_list(self.layer_dic_list)


    def forward(self, input):
        layer_dic_list = self.layer_dic_list[1:]
        module_list = self.module_list
        feature_map_list = []
        dtctn_exists = False
        
        for index, layer_dic in enumerate(layer_dic_list):
            
            if layer_dic[LAYER_TYPE] == "convolutional":
                output = module_list[index](input)

            elif layer_dic[LAYER_TYPE] == "shortcut":
                from_layer = int(layer_dic["from"])
                abs_shrtct_layer = index + from_layer
                output = feature_map_list[index - 1] + feature_map_list[abs_shrtct_layer]

            elif layer_dic[LAYER_TYPE] == "upsample":
                output = module_list[index](input)

            elif layer_dic[LAYER_TYPE] == "route":
                layers = layer_dic["layers"]
                
                if "," in layers:
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

                height = int(self.net_info["height"])
                anchor_str = layer_dic["anchors"].split(",")
                mask = layer_dic["mask"].split(",")

                anchors = get_anchors(anchor_str, mask)
                
                output = transform_yolo_output(input, anchors, height)
                
                if dtctn_exists:
                    detection_tensor = torch.cat((detection_tensor, output), 1)
                else:
                    detection_tensor = output
                    dtctn_exists = True
            feature_map_list.append(output)
            input = output
            
        return detection_tensor

    # The load_weights functions has been copied as it is from Ayoosh kathuria's blog.
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
            module_type = self.layer_dic_list[i + 1][LAYER_TYPE]
        
            #If module_type is convolutional load weights
            #Otherwise ignore.
                
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.layer_dic_list[i+1]["batch_normalize"])
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

def analyze_transactions(img, cnf_thres = 0.5, iou_thres = 0.4):
    img = img[0]
    img = img[img[:, 4] > cnf_thres]
    
    # no detections
    if img.size()[0] == 0:
        return 0

    # convert bx, by, bw, bh into bx1, by1, bx2, by2
    boxes = img.new(img.shape)
    boxes[:,0] = img[:, 0] - img[:, 2]/2
    boxes[:,1] = img[:, 1] - img[:, 3]/2
    boxes[:,2] = img[:, 0] + img[:, 2]/2
    boxes[:,3] = img[:, 1] + img[:, 3]/2
    img[:, :4] = boxes[:,:4]
    

    max_values, class_values = torch.max(img[:,5:], 1)

    img = torch.cat((img[:, :5], max_values.float().unsqueeze(1), class_values.float().unsqueeze(1)), 1)
    
    ## at this point, the img tensor has 7 values in each row: bx1, by1, bx2, by2, confidence, cls_confidence, class
    dtctn_tnsr_exsts = False
    classes = torch.unique(img[:, 6])
    for cls in classes:
        # retrieve all the rows which correspond to class cls
        cls_tensor = img[torch.where(img[:, 6] == cls)]

        # sort cls_tensor according to max confidence
        cls_tensor = cls_tensor[cls_tensor[:,5].sort(descending = True)[1]]
           
        # box_iou takes tensors which have only 4 columns
        iou_tensor = tvo.box_iou(cls_tensor[:,:4], cls_tensor[:,:4])
    
        rejected_indices = []
        detected_indices = []
        #TODO: have a helper function to generate an image with all bbs drawn at this stage
        for row in range(iou_tensor.size(0)):
            if row in rejected_indices:
                continue
            exceeding_thres_tensor = torch.where(iou_tensor[row] > iou_thres)[0]
            rejected_indices.extend(exceeding_thres_tensor.tolist())
            detected_indices.append(row)
        
        if dtctn_tnsr_exsts:
            detection_tensor = torch.cat((detection_tensor, cls_tensor[detected_indices]), 0)
                
        else:
            detection_tensor = cls_tensor[detected_indices]
            dtctn_tnsr_exsts = True
    try:
        return detection_tensor
    except:
        return 0


def detect(folder, image):
    img = os.path.join(folder, image)
    img_tensor = image_to_tensor(img)

    net = Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")
    net.eval()
    with torch.no_grad():
        detections = net(img_tensor)
        detections = analyze_transactions(detections, cnf_thres = 0.5, iou_thres = 0.4)
        classes = read_classes("assets/coco.names")
        draw_rectangle(img, detections, classes)



parser = argparse.ArgumentParser()
parser.add_argument("--folder", default = "images", type=str)
parser.add_argument("--name", default = "eagle.jpg", type=str)
args = parser.parse_args()
image_folder = args.folder
image = args.name

detect(image_folder, image)

