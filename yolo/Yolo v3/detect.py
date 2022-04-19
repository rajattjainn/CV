import argparse
import os

import torch

import neural_net
import utils



def detect(detect_loader):
    net = neural_net.Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")
    net.eval()
    for index, features in enumerate(detect_loader):
        with torch.no_grad():
            detections = net(features)
            print (len(detections))
            print (detections[0].size())
                

    pass

def detect_old(folder, image):
    img = os.path.join(folder, image)
    img_tensor = utils.image_to_tensor(img)

    net = neural_net.Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")
    net.eval()
    with torch.no_grad():
        print ("\ndetections")
        detections = net(img_tensor)
        print (len(detections))
        print (detections[0].size())
        
        detections = neural_net.analyze_transactions(detections, cnf_thres = 0.5, iou_thres = 0.4)
        classes = neural_net.read_classes("assets/coco.names")
        neural_net.draw_rectangle(img, detections, classes)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default = "images", type=str)
parser.add_argument("--name", default = "eagle.jpg", type=str)
args = parser.parse_args()
image_folder = args.folder
image = args.name

detect_old(image_folder, image)



# detect_label_path = "/Users/Jain/code/learnML/yolo/Yolo v3/images"
# detect_loader = utils.get_dataloader(detect_label_path, None)

# detect(detect_loader)