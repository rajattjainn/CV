import numpy as np
import torch

def transform_prediction(predictions, anchors, num_classes, input_height):
    """
    """
    batch_size = predictions.size(0)
    out_grid_size = predictions.size(2)
    num_anchors = len(anchors)
    bbox_attributes_len = 5 + num_classes
    stride = input_height // out_grid_size

    # The first task is to convert the predictions in a more operable format.
    # The format of predictions is = batch_size x (B * (5 + C)) x out_grid_size x out_grid_size.
    # B is the number of bounding boxes (or anchors), C is the number of classes. The second dim
    # has one anchor box stacked over another. This will make it difficult to perform mathematical
    # operations, hence we need to reshape the tensor so that each item is easily accessible.

    # First of all we remove one trailing dimension of the tensor. This means the last dimension 
    # has a size of (out_grid_size * out_grid_size). After that we swap 2nd and 3rd dimensions. 
    # Next, in the third dimension, we decouple various anchor boxes from each other. Before this 
    # operation, the third dimension can be visualised as a vertical array with each row 
    # representing a cell of the detection map having the data for three anchor boxes. We'll 
    # unfold each row into B rows so that each row has data of only one anchor box

    predictions = predictions.view(batch_size, bbox_attributes_len * num_anchors, out_grid_size * out_grid_size)
    predictions = predictions.transpose(1,2).contiguous()
    predictions = predictions.view(batch_size, out_grid_size * out_grid_size * num_anchors, bbox_attributes_len)

    # Now in the last dimension, the first five elements correspond to tx, ty, tw, th, and 
    # p (confidence score). Post that, the data corresponds to confidence for each class.

    # We'll apply sigmoid function to tx, ty and p first.

    predictions[:, :, 0] = torch.sigmoid(predictions[:, :, 0])
    predictions[:, :, 1] = torch.sigmoid(predictions[:, :, 1])
    predictions[:, :, 4] = torch.sigmoid(predictions[:, :, 4])

    # Next, we need to add offset to each item of the last dimension according to it's relevant 
    # position in the output grid. For clarity, the coordinates in the yolo layer has the following 
    # operation performed.
    # bx = sigmoid(tx) + cx; by = sigmoid(ty) + cy

    mesh = np.arange(out_grid_size)
    a, b = np.meshgrid(mesh, mesh)

    x_offset = torch.Tensor(a)
    y_offset = torch.Tensor(b)

    x_offset = x_offset.view(-1, 1)
    y_offset = y_offset.view(-1, 1)

    x_y_offset = torch.cat((y_offset, x_offset), 1)
    x_y_offset = x_y_offset.repeat(1, num_anchors)
    x_y_offset = x_y_offset.view(-1, 2).unsqueeze(0)

    print ("\n\npredictions and x_y_offset")
    print (predictions.size())
    print (x_y_offset.size())
    predictions[:, :, 0:2] += x_y_offset

    # performing operations on the tw and th values.
    # bh = ph * e^th
    # bw = pw * e^tw
    # pw and ph are 3rd and 4th values in each row of the 
    # last dimension of prediction tensor. Beofore we do the 
    # above calculations, we need to convert anchors into 
    # a tensor of matching dimensions as well.

    anchor_tensor = torch.Tensor(anchors)
    anchor_tensor = anchor_tensor.repeat(out_grid_size *out_grid_size, 1).unsqueeze(0)
    predictions[:, :, 2:4] = (torch.exp(predictions[:, :, 2:4])) * anchor_tensor
    print ("\n\nprediction size after mul")
    print (predictions.size())

    predictions[:, :, 0:4] = predictions[:, :, 0:4] * stride
    print ("/n/nafter stride mul")
    print (predictions.size())

    return predictions
