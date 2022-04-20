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


    
def train():
    EPOCHS = 5
    EPOCHS = 1
    net = neural_net.Yolo3("assets/config.cfg")
    net.load_weights("assets/yolov3.weights")

    batch_size = len(train_loader)
    optimizer = optim.SGD(net.parameters(), 
                        lr=0.001/batch_size, momentum=0.9, 
                        dampening=0, weight_decay=.0005*batch_size)

    for epoch in range (EPOCHS):
        prdctn_exists = False
        target_labels = []
        net.train()
        print ("\n\nEpoch: " + str(epoch))

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

        loss.backwards()
        optimizer.step()

train()