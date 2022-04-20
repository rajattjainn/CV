import torch
import torchvision.ops as tvo
import numpy as np
import torch.nn as nn

import utils as utils
import neural_net

train_label_path = "/Users/Jain/code/cloned/ultralytics/coco128/train/labels/train2017"
train_image_path = "/Users/Jain/code/cloned/ultralytics/coco128/train/images/train2017"
train_loader = utils.get_dataloader(train_image_path, train_label_path)

eval_label_path = "/Users/Jain/code/cloned/ultralytics/coco128/eval/labels/train2017"
eval_image_path = "/Users/Jain/code/cloned/ultralytics/coco128/eval/images/train2017"
eval_loader = utils.get_dataloader(eval_image_path, eval_label_path)


def individual_loss(predicted_tensor, target_labels):
    """
    Calculate loss between prediction and targets corresponding to one image.
    """
    # Create a target tensor from the string of target labels provided.
    # The number of targets would be less than total number of predictions
    rows = target_labels.count('\n')
    label_array = np.fromstring(target_labels, sep=' ').reshape(rows, -1)
    target_tensor = torch.from_numpy(label_array)
    
    # Calculate iou between the target tensor and predicted tensor. This would be a "m x n" matrix,
    # where "m" is the number of targets and "n" is the number of predictions by the network.
    predicted_tensor_box = predicted_tensor[:, 1:5] / 416 # the predicted_tensor has already been scaled to actual dimensions
    target_tensor_box = target_tensor[:, 1:5]
    iou_tensor = tvo.box_iou(target_tensor_box, predicted_tensor_box)

    # Get the best predicted boxes, i.e. predicted boxes which are closest to ground truth boxes.
    # Number of best predicted boxes = number of target boxes
    _, best_indices = torch.max(iou_tensor, 1)
    predicted_gt_boxes = predicted_tensor_box[best_indices]
    
    # Calculate coordinate loss. Coordinate loss is calculated only for the best boxes.
    predicted_gt_boxes = predicted_gt_boxes.float()
    target_tensor_box = target_tensor_box.float()
    coord_loss = nn.BCELoss(reduction="sum")(predicted_gt_boxes[0:2], target_tensor_box[0:2]) + \
        nn.MSELoss(reduction="sum")(predicted_gt_boxes[2:4], target_tensor_box[2:4])

    # First we retrieve the classes for predicted best (predicted gt) boxes.
    # Then we calculate class loss, only for the best boxes. 
    predictedgt_cls_tensor, _ = torch.max(predicted_tensor[best_indices, 5:], 1)
    class_loss = nn.BCEWithLogitsLoss(reduction="sum")(predictedgt_cls_tensor, target_tensor[:, 0])

    # Retrieve the max of iou values between each predicted tensor and all target boxes.
    # Use these values and confidence value predicted by the network to calculate confidence loss
    prediction_gt_iou_tensor, _ = torch.max(iou_tensor, 0)
    prediction_gt_iou_tensor = prediction_gt_iou_tensor.float()
    conf_loss = nn.BCELoss(reduction="sum")(prediction_gt_iou_tensor, predicted_tensor[:, 5])
    
    # Total loss for a particular prediction is summation of all the above 3 losses.
    loss = coord_loss + class_loss + conf_loss 

    return loss


def calculate_loss(predicted_tensor, target_labels):
    """
    Calculate the loss between predictions and targets in one batch.
    """
    assert len(predicted_tensor) == len(target_labels)

    total_loss = 0
    for i in range(len(predicted_tensor)):
        loss = individual_loss(predicted_tensor[i], target_labels[i])
        total_loss =+ loss
        
    return total_loss/len(predicted_tensor)

    
def train():
    EPOCHS = 5
    EPOCHS = 1
    net = neural_net.Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")

    for epoch in range (EPOCHS):
        prdctn_exists = False
        target_labels = []
        net.train()
        print ("\n\nEpoch: " + str(epoch))

        # grad = 0
        for _, (features, labels) in enumerate(train_loader):
                detections = net(features)
                if prdctn_exists:
                    predicted_tensor = torch.cat((predicted_tensor, detections), 0)
                else:
                    predicted_tensor = detections
                    prdctn_exists = True
                target_labels.extend(list(labels))

        # predicted_tensor = torch.load("detection.pt")
        loss = calculate_loss(predicted_tensor, target_labels)

        # loss.backwards()
        # optim.step()

train()