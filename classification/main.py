import torch
from torchvision import models
from PIL import Image
import numpy as np
from collections import OrderedDict

from zmq import device
from dataset import CustomDataSet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

# create your model 
model = models.resnet18(pretrained=False)

# hyperparameter setting
LR = 1e-5
EPOCH = 200
device = "cuda"
batch_size = 64

# log file
logger.add("output/single_moco_correct/train.log")

# load pre-trained model
pretrained_dict = torch.load("checkpoints/moco_same_backbone.pth")
simSim_dict = torch.load("checkpoints/simsiam_2layer_backbone.pth")
model_dict = model.state_dict()

temp_dict = OrderedDict()
cot = 0
for (k, v), (k_, v_) in zip(pretrained_dict['state_dict'].items(), simSim_dict['state_dict'].items()):
    # name = k.replace("backbone.", "")
    name = k
    if "fc" not in name and cot < 30:
        temp_dict[name] = v
    elif "fc" not in name:
        temp_dict[name] = v

model_dict.update(temp_dict)
model.load_state_dict(model_dict)

model.fc = nn.Linear(in_features=512, out_features=2)

# load data

train_ds = CustomDataSet(mode="train")
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)

test_ds = CustomDataSet(mode="test")
test_loader = DataLoader(test_ds, shuffle=True, batch_size=batch_size)

# optimization

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model = model.to(device)

# training

for epoch in tqdm(range(EPOCH)):
    model.train()
    logger.info(f"start train at epoch {epoch}")
    total_loss = 0
    acc_train = 0
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        y = y.long()

        pred = model(x)
        out = torch.argmax(torch.sigmoid(pred), dim=1)
        acc_train += torch.sum(y.squeeze() == out) / len(out)
        loss = criterion(pred, y.squeeze())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    logger.info(f"training loss is {total_loss / (idx + 1)}")
    logger.info(f"train accuracy is {acc_train / (idx + 1)}")

    logger.info(f"evl at epoch {epoch}")

    # eval
    model.eval()
    test_loss = 0
    acc_test = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            y = y.long()

            pred = model(x)
            loss = criterion(pred, y.squeeze())
            out = torch.argmax(torch.sigmoid(pred), dim=1)
            acc_test += torch.sum(y.squeeze() == out) / len(out)

            test_loss += loss.item()
    logger.info(f"test loss is {test_loss / (idx + 1)}")
    logger.info(f"test accuracy is {acc_test / (idx + 1)}")
    model.train()
