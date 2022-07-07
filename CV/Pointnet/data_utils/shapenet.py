import os
import json

import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch

class Shapenet(Dataset):
    def __init__(self, root_dir, split_file) -> None:
        super().__init__()
        self.root_dir = root_dir
        
        map_file = os.path.join(root_dir, "synsetoffset2category.txt")
        with open(map_file) as mf:
            lines = mf.readlines()
            self.obj_fldr_map_dict = {i.split()[1]: i.split()[0] for i in lines}
        # print (self.obj_fldr_map_dict)

        cur_file_root_pth = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clss_map_file = os.path.join(cur_file_root_pth, "assets", "class_mapping.txt")
        with open(clss_map_file) as cmf:
            lines = cmf.readlines()
            self.classes = {i.split()[1]: i.split()[0] for i in lines}
            # print (self.classes)

        split_file = json.load(open(os.path.join(root_dir, 
                    "train_test_split", "shuffled_{}_file_list.json".format(split_file))))
        self.data = [(path.split("shape_data/")[1]) for path in split_file]
        

    def __getitem__(self, idx):
        cur_data = self.data[idx].split("/")
        file_path = os.path.join(self.root_dir, cur_data[0], "points", cur_data[1] + ".pts")
        with open(file_path) as pt_dt:
            rows = [row.split() for row in pt_dt]

        pt_array = np.array(rows).astype("float32")
        pt_tensor = torch.tensor(pt_array)

        pt_folder = cur_data[0]
        pt_type = self.obj_fldr_map_dict[pt_folder]
        pt_cls = self.classes[pt_type]
        return pt_tensor, pt_cls

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    root_dir = "/Users/eureka/Desktop/3D_Data/shapenetcore_partanno_segmentation_benchmark_v0"
    shapenet = Shapenet(root_dir, "train")

    loader = DataLoader(shapenet, batch_size = 1)
    for index, (pt_tensor, pt_cls) in enumerate(loader):
        if index == 0:
            raise Exception ("Uh oh")

