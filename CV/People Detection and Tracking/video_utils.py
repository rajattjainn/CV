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
    weightsPath = os.path.join(current_dir, "assets", "yolov3.weights")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net

def detect_object(net, frame, conf_thres, nms_thres):
    
    oln = net.getUnconnectedOutLayersNames()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (main.IMAGE_WIDTH, main.IMAGE_HEIGHT),
	swapRB=True, crop=False)
    
    net.setInput(blob)
    layer_outputs = net.forward(oln)

    layer_outputs = np.concatenate(layer_outputs)    
    thres_idx = layer_outputs[:,4] > conf_thres
    relevant_outputs = layer_outputs[thres_idx]
    
    (H,W) = frame.shape [:2]
    relevant_outputs[:,0] = relevant_outputs[:,0] * W
    relevant_outputs[:,1] = relevant_outputs[:,1] * H 
    relevant_outputs[:,2] = relevant_outputs[:,2] * W
    relevant_outputs[:,3] = relevant_outputs[:,3] * H

    boxes = np.zeros((len(relevant_outputs), 4))
    boxes[:, 0] = relevant_outputs[:,0] - relevant_outputs[:,2]/2
    boxes[:, 1] = relevant_outputs[:,1] - relevant_outputs[:,3]/2
    boxes[:, 2] = relevant_outputs[:,2]
    boxes[:, 3] = relevant_outputs[:,3]
    
    max_conf_index = np.argmax(relevant_outputs[:,5:], axis=1)
    class_conf = np.amax(relevant_outputs[:,5:], axis=1)

    
    indexs = cv2.dnn.NMSBoxes(boxes, class_conf, conf_thres, nms_thres)
    print (len(indexs))

	
		
    # outputs = []
    # # Filter all the outputs according to conf_thres.
    # # Combine all the outputs in one array
    # print (layer_outputs[0].shape)
    # for lo in layer_outputs:
    #     thres_idx = lo[:,4] > conf_thres
    #     lo = lo[thres_idx]
    #     print ("lo.shape")
        
    #     print (lo.shape)
    #     outputs.extend(lo)

    # print(outputs[0].shape)
    # print (len(outputs))
    # output = np.concatenate(outputs, axis=1)
    # print(output.shape)
    # boxes = np.zeros((len(output), 4))
    # print ("boxes.shape")
    # print (boxes.shape)
    
    # confidences = []
    # labels = []
    
    

    # # for out in outputs:
        
    # print (len(outputs))
    # print (outputs[0].shape)
    # confidences = outputs[:, 4]
    
    
    
    
    
    
    
    
