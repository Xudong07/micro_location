#-*- coding:utf-8 -*-
import os
import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision
from load_data import LoadDataPredict
from  get_model import getM
from engine import train_one_epoch, evaluate
import utils
import pandas as pd
import datetime
import time
import torch
import torch.utils.data
from torch import nn


model_dir = './model/mode_9.pth'
data_dir = './test/'
os.environ['CUDA_VISIBLE_DEVICES']='2'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
weight=(1.0,1.0,1.0,100.0)
print(device)
datasets_pred = LoadDataPredict(data_dir)
model = getM(dropout=None, weight=weight)
model.to(device)
checkpoint = torch.load(model_dir, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
lentest = len(datasets_pred)
print('there are {} sample to be predicted'.format(lentest))
name_class=[]
for i in range(lentest):
    #print(i)
    sample = datasets_pred[i]
    sample = sample.unsqueeze(0)
    sample=sample.to(device)
    #print(sample.shape)
    prediction = model(sample)
    print(prediction)
    #prediction=prediction[0]

print("That's it!")




