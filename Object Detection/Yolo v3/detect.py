import argparse
import os

import torch

import neural_net
import utils


def detect(image_dir_path):
    detect_loader = utils.get_dataloader(image_dir_path, shuffle=False)
    net = neural_net.Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")
    
    net.eval()
    dtctn_exists = False
    for _, features in enumerate(detect_loader):
        with torch.no_grad():
            detections = net(features)
            if dtctn_exists:
                detection_tensor = torch.cat((detection_tensor, detections), 0)
            else:
                detection_tensor = detections
                dtctn_exists = True

    images =  [f for f in os.listdir(image_dir_path) if f.endswith(('.jpg', '.jpeg', 'png'))]
    
    # Loop over each detection. One detection corresponds to one image
    for det_ind in range(len(detection_tensor)):
        det = detection_tensor[det_ind]
        det = neural_net.analyze_detections(det, cnf_thres = 0.5, iou_thres = 0.4)
        if isinstance(det, int):
            continue
        classes = utils.read_classes("assets/coco.names")
        
        img = os.path.join(image_dir_path, images[det_ind])
        utils.draw_rectangle(img, det, classes)



image_dir_path = "/Users/Jain/code/cloned/ultralytics/coco128/train/images/train2017"
detect(image_dir_path)
