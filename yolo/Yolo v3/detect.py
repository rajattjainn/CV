import argparse
import os

import torch

import neural_net
import utils



def detect(image_dir_path):
    detect_loader = utils.get_dataloader(image_dir_path, None, shuffle=False)
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
    for det_ind in range(len(detection_tensor)):
        det = detection_tensor[det_ind]
        det = neural_net.analyze_transactions(det, cnf_thres = 0.5, iou_thres = 0.4)
        if isinstance(det, int):
            continue
        classes = neural_net.read_classes("assets/coco.names")
        
        img = os.path.join(image_dir_path, images[det_ind])
        neural_net.draw_rectangle(img, det, classes)



image_dir_path = "/Users/Jain/code/learnML/yolo/Yolo v3/images"
detect(image_dir_path)


# def detect_old(folder, image):
#     img = os.path.join(folder, image)
#     img_tensor = utils.image_to_tensor(img)

#     net = neural_net.Yolo3("assets/config.cfg")
#     net.load_weights("assets/yolov3.weights")
#     net.eval()
#     with torch.no_grad():
#         print ("\ndetections")
#         detections = net(img_tensor)
#         print (len(detections))
#         print (detections[0].size())
        
#         detections = neural_net.analyze_transactions(detections, cnf_thres = 0.5, iou_thres = 0.4)
#         classes = neural_net.read_classes("assets/coco.names")
#         neural_net.draw_rectangle(img, detections, classes)

# parser = argparse.ArgumentParser()
# parser.add_argument("--folder", default = "images", type=str)
# parser.add_argument("--name", default = "eagle.jpg", type=str)
# args = parser.parse_args()
# image_folder = args.folder
# image = args.name

# detect_old(image_folder, image)
