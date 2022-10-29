#!/usr/bin/python
# -*- coding: UTF-8 -*-


from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_arguments("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path of images")
parser.add_argument("--batch_size", type=int, default=16, help="num of each epoch")
parser.add_argument("--model_config_path", type=str, default="config/yolov3.cfg", help="config of net")
parser.add_argument("--data_config_path", type=str, default="config/coco.names", help="name of classes")
parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path of weights")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="objects classes")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresthrold for class")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to train")
parser.add_argument("--img_size", type=int, default=416, help="size of image")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval being ")
parser.add_argument("--use_cuda", type=bool, default=True, help="wheather to use cuda")
opt = parser.parse_args([])
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda
os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

classes = load_classes(opt.class_path)

data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]

hyperparams = parse_model_config(opt.data_config_path)
learning_rates = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

model = Darknet(opt.model_config_path)

model.apply(weights_init_normal)

if cuda:
    model = model.cuda()

model.train()

dataloader = torch.utils.data.DataLoader(ListDataset(train_path), batch_size=opt.batch_size, shuffle=False,
                                         num_workers=opt.n_cpu)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
for epoch in range(opt.epochs):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        optimizer.step()
        print("[Epoch %d/%d, Batch %d/%d][Losses: x%f, y%f, w%f, h%f, conf%f, cls%f, total%f, recall%.5f, "
              "precision%.5f] "
              % (epoch, opt.epochs, batch_i, len(dataloader), model.losses["x"], model.losse["y"], model.losses["w"],
                 model.losses["h"],
                 model.losses["conf"], model.losses["cls"], loss.item(), model.losses["recall"],
                 model.losses["precision"],))

        model.seen += imgs.size(0)
        if epoch % opt.checkpoint_interval == 0:
            model.save_weights("%s/%d.weights") % (opt.checkpoint_dir, epoch)
