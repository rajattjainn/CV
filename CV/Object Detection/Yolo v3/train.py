from datetime import datetime

import torch
import torch.optim as optim

import utils as utils
import neural_net

train_label_path = "/Users/Jain/code/cloned/ultralytics/coco128/train/labels/train2017"
train_image_path = "/Users/Jain/code/cloned/ultralytics/coco128/train/images/train2017"
train_loader = utils.get_dataloader(train_image_path, train_label_path)

eval_label_path = "/Users/Jain/code/cloned/ultralytics/coco128/eval/labels/train2017"
eval_image_path = "/Users/Jain/code/cloned/ultralytics/coco128/eval/images/train2017"
eval_loader = utils.get_dataloader(eval_image_path, eval_label_path)

log_file = "TrainingLog_" + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S') + ".txt"

def save_model_weights(epoch, model):
    torch.save(model.state_dict(), "model_weights_" + str(epoch) + ".pth")

def train():
    EPOCHS = 10
    net = neural_net.Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")

    batch_size = len(train_loader)
    optimizer = optim.SGD(net.parameters(), 
                        lr=0.001/batch_size, momentum=0.9, 
                        dampening=0, weight_decay=.0005*batch_size)
    
    train_loss = []
    eval_loss = []

    for epoch in range (EPOCHS):
        with open(log_file, "a") as f:
            f.write("Start Time: " + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S'))
            f.write("\n")
            f.write("Epoch: " + str (epoch))
            f.write("\n")
            
        net.train()
       
        train_running_loss = 0
        for _, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            detections = net(features)
            
            loss = utils.calculate_loss(detections, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()        
        
        train_epch_loss = train_running_loss/len(train_loader)    
        with open(log_file, "a") as f:
            f.write("Epoch {}, Train Loss: {}".format(epoch, train_epch_loss))
        train_loss.append(train_epch_loss)
        
        net.eval()
        eval_running_loss = 0
        with torch.no_grad():
            for _, (eval_features, eval_labels) in enumerate(eval_loader):
                eval_detections = net(eval_features)
                loss = utils.calculate_loss(eval_detections, eval_labels)
                eval_running_loss += loss.item()
            
        eval_epch_loss = eval_running_loss/len(eval_loader)    
        with open(log_file, "a") as f:
            f.write("Epoch {}, Eval Loss: {}".format(epoch, eval_epch_loss))
        eval_loss.append(eval_epch_loss)
  
        with open(log_file, "a") as f:
            f.write("End Time: " + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S'))
            f.write ("\n\n")

        save_model_weights(epoch, net)

train()