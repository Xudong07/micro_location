#-*- coding:utf-8 -*-
import os
import numpy as np
import torch 
from PIL import Image
from torch.utils.data import Dataset
#import xmltodict
import matplotlib.pyplot as plt
import torchvision.transforms as T
#from data_aug import DataAugmentForObjectDetection
from scipy.io import loadmat

class LoadDataPredict(Dataset):
    def __init__(self,root):
        self.root = root
        self.imgf = list(sorted(os.listdir(os.path.join(root,'data'))))
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root,'data',self.imgf[index])
        #read data
        img = loadmat(img_path)['wave']
        # read targe
        img = torch.tensor(img, dtype=torch.float32)
        return img
    def __len__(self):
        return len(self.imgf)

class LoadData(Dataset):
    def __init__(self,root,train=True):
        self.root = root
        self.imgf = list(sorted(os.listdir(os.path.join(root,'data'))))
        self.targetf = list(sorted(os.listdir(os.path.join(root,'label'))))
        self.train = train
    def __getitem__(self, index):
        img_path = os.path.join(self.root,'data',self.imgf[index])
        label_path = os.path.join(self.root,'label', self.targetf[index])
        #read data
        img = loadmat(img_path)['wave']
        # read target
        target = loadmat(label_path)['label']
        target = target.reshape(-1,)
        img = torch.tensor(img, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return img, target
    def __len__(self):
        return len(self.imgf)

def imshow(img,target):
    print(img.shape)
    print(target.shape)
    print(type(img), type(target))
    plt.plot(img[0,0,:])
    plt.show()

if __name__=='__main__':
    datasets = LoadData('./',train=True )
    lend = len(datasets)
    print(lend)
    for i in range(0,3,1):
        print(i)
        img, target = datasets[i]
        print(img.shape, target.shape)
        #imshow(img,target)



