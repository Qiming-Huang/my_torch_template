import torch
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import numpy as np
import os 

import matplotlib.pyplot as plt

import pickle



class CustomDataSet(Dataset):
    def __init__(self, mode):
        self.img_list, self.labels = self._init()
        self.train_img_list = self.img_list[:int(0.8 * len(self.img_list))]
        self.test_img_list = self.img_list[int(0.8 * len(self.img_list)):]

        self.train_label = self.labels[:int(0.8 * len(self.labels))]
        self.test_label = self.labels[int(0.8 * len(self.labels)):]

        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train_img_list)
        else:
            return len(self.test_img_list)

    def __getitem__(self, index):
        
        if self.mode == "train":

            label = self.train_label[index]
            img = Image.open(self.train_img_list[index]).convert("L")
            img = img.resize((512, 512))
            img = np.array(img) / 255
            img = torch.as_tensor(img, dtype=torch.float32)
            img = img.unsqueeze(0)
            img = img.repeat(3, 1, 1)


            # img = img.permute((2, 0, 1))
            label = torch.as_tensor(int(label))


            return img, label
        else:
            label = self.test_label[index]

            img = Image.open(self.test_img_list[index]).convert("L")
            img = img.resize((512, 512))
            img = np.array(img) / 255
            img = torch.as_tensor(img, dtype=torch.float32)
            img = img.unsqueeze(0)
            img = img.repeat(3, 1, 1)


            # img = img.permute((2, 0, 1))
            label = torch.as_tensor(int(label))


            return img, label


    def _init(self):
        image_list = []
        label_list = []
        labels = []
        for root, dirs, files in os.walk("/data/qiming/music/codes/idea/ChinaSet_AllFiles/CXR_png"):
            for file in files:
                if ".png" in file:
                    image_list.append(os.path.join(root, file))
        for root, dirs, files in os.walk("/data/qiming/music/codes/idea/ChinaSet_AllFiles/ClinicalReadings"):
            for file in files:
                if ".txt" in file:
                    label_list.append(os.path.join(root, file))
        
        image_list = sorted(image_list)
        label_list = sorted(label_list)
        for i in label_list:
            labels.append(i.split("_")[-1][0])

        return image_list, labels

# save yuur dataset inti .pkl files can faster your data loading while use extra storage.

class Scene_pickle(Dataset):
    def __init__(self, mode):
        self.mode = mode
        res = open('/data/qiming/music/dataset/Scene/scene_train.pkl','rb')
        data = pickle.load(res)

        self.train_imgs = data['imgs']
        self.train_labels = data['labels']

        res = open('/data/qiming/music/dataset/Scene/scene_test.pkl','rb')
        data = pickle.load(res)

        self.test_imgs = data['imgs']
        self.test_labels = data['labels']

    def __len__(self):
        if self.mode == "train":
            return len(self.train_imgs)
        else:
            return len(self.test_imgs)

    def __getitem__(self, index):
        
        if self.mode == "train":

            return self.train_imgs[index], self.train_labels[index]
        else:
            return self.test_imgs[index], self.test_labels[index]
