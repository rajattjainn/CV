import os
import json

import torch
from torch import nn
from torch.utils.data import Dataset

class Shapenet(Dataset):
    def __init__(self, root_dir, split_file) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split_file = split_file

        map_file = os.path.join(root_dir, "synsetoffset2category.txt")
        with open(map_file) as mf:
            lines = mf.readlines()
            self.cls_fldr_map_dict = {i.split()[1]: i.split()[0] for i in lines}
        folders = list(self.cls_fldr_map_dict.keys())
        self.classes = {folders[i]:i for i in range((len(folders)))}

        split_file = json.load(open(os.path.join(root_dir, 
                    "train_test_split", "shuffled_{}_file_list.json".format(split_file))))
        self.data = [(path.split("shape_data/")[1]) for path in split_file]
        

if __name__ == "__main__":
    root_dir = "/Users/eureka/Desktop/3D_Data/shapenetcore_partanno_segmentation_benchmark_v0"
    shapenet = Shapenet(root_dir, "test")