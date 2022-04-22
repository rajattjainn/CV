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

    for epoch in range (EPOCHS):
        with open(log_file, "a") as f:
            f.write("Epoch: " + str (epoch))
            f.write("\n")
            f.write("Start Time: " + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S'))
            print ("\n")

        prdctn_exists = False
        target_labels = []
        net.train()

        optimizer.zero_grad()
        for _, (features, labels) in enumerate(train_loader):
                detections = net(features)
                if prdctn_exists:
                    predicted_tensor = torch.cat((predicted_tensor, detections), 0)
                else:
                    predicted_tensor = detections
                    prdctn_exists = True
                target_labels.extend(list(labels))

        # predicted_tensor = torch.load("detection.pt")
        loss = utils.calculate_loss(predicted_tensor, target_labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss = loss.item()
        
        
        eval_prdctn_exists = False
        eval_target_labels = []
        net.eval()
        with torch.no_grad:
            for _, (eval_features, eval_labels) in enumerate(eval_loader):
                eval_detections = net(eval_features)
                if eval_prdctn_exists:
                    eval_predicted_tensor = torch.cat((eval_predicted_tensor, eval_detections), 0)
                else:
                    eval_predicted_tensor = eval_detections
                    eval_prdctn_exists = True
                eval_target_labels.extend(list(eval_labels))
            
            eval_loss = utils.calculate_loss(eval_predicted_tensor, eval_target_labels)
            epoch_eval_loss = eval_loss.item()
        
        with open(log_file, "a") as f:
            f.write("\n")
            f.write("Train Loss: " + str (epoch_train_loss))
            f.write("\n")
            f.write("Eval Loss: " + str (epoch_eval_loss))
            f.write ("\n")
            f.write("Epoch: " + str (epoch))
            f.write("\n")
            f.write("End Time: " + datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S'))
            f.write ("\n\n")

        save_model_weights(epoch, net)
train()