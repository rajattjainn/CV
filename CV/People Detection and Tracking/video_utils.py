import os
import time

import cv2
import numpy as np

import main

def get_yolov3_network():
    """
    Return an object of cv2.dnn.Net class. The object will 
    correspond to a Yolo v3 network.
    """
    current_dir = os.path.abspath(os.getcwd())
    configPath = os.path.join(current_dir, "assets", "yolov3.cfg")
    weightsPath = os.path.join(current_dir, "assets", )
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    return net

def detect_object(net, frame):
    
    oln = net.getUnconnectedOutLayersNames()
    ims = cv2.resize(frame, (main.IMAGE_WIDTH, main.IMAGE_HEIGHT))
    blob = cv2.dnn.blobFromImage(ims, 1 / 255.0,
	swapRB=True, crop=False)
    
    net.setInput(blob)
    layer_outputs = net.forward(oln)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
