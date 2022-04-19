import os

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from datasets import ObjectDataSet
import utils as utils
import neural_net

train_label_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/train/labels/train2017"
train_image_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/train/images/train2017"
train_loader = utils.get_dataloader(train_image_path, train_label_path)

eval_label_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/eval/labels/train2017"
eval_image_path = "/Users/Jain/code/cloned/ultralytics/coco128_backup/eval/images/train2017"
eval_loader = utils.get_dataloader(eval_image_path, eval_label_path)

