#-*- coding:utf-8 -*-
import torch
import os
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=(1,1)):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=(1,1)):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=(1,1), downsample=None):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        #each instance has their own parameters
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out +=identity
        out = self.relu(out)
        return out





class Model(nn.Module):
    def __init__(self, block,dropout=None,weight=(1.0,1.0,1.0,100.0)):
        super(Model,self).__init__()
        self.dropout = dropout
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        #for bn, bias can be set false
        self.conv1 = nn.Conv2d(3,self.inplanes, kernel_size=7, stride=(1,2), padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #your imge is 12*1000, don't pool in the the dimension of 12
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2),padding=(0,1))


        self.layer1 = self._make_layer(block,64,2)
        self.layer2 = self._make_layer(block,128,2,stride=2)
        self.layer3 = self._make_layer(block,256,2, stride=2)
        self.layer4 = self._make_layer(block,512,2,stride=2)
        
        if dropout is not None:
            print('using dropout')
            self.regression = nn.Sequential(
                nn.Linear(512*12*32, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(1024,27)
            )
        else:
            self.regression = nn.Sequential(
                nn.Linear(512*12*32, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,27)
            )
        self._init_w()
        self.weight = weight
        print('weight for location, depth, vel, vel_ratio', self.weight)
        


    def compute_loss(self, out, target):
        loss_loc = self.weight[0] * torch.mul(out[:,:3]-target[:,:3], out[:,:3]-target[:,:3])
        loss_depth = self.weight[1] * torch.mul(out[:,3:11]-target[:,3:11], out[:,3:11]-target[:,3:11])
        loss_vel = self.weight[2] * torch.mul(out[:,11:19]-target[:,11:19], out[:,11:19]-target[:,11:19])
        loss_ratio = self.weight[3] * torch.mul(out[:,19:]-target[:,19:], out[:,19:]-target[:,19:])
        num_batch = len(target)
        loss = {}
        loss['loc_loss'] = loss_loc.sum()/num_batch
        loss['depth_loss'] = loss_depth.sum()/num_batch
        loss['vel_loss'] = loss_vel.sum()/num_batch
        loss['ratio_loss'] = loss_ratio.sum()/num_batch
        return loss

    def forward(self,x, target=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x=self.maxpool(x)
        #print(x.shape)
        x=self.layer1(x)
        #print(x.shape)
        x=self.layer2(x)
        #print(x.shape)
        x=self.layer3(x)
        #print(x.shape)
        x=self.layer4(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        out = self.regression(x)
        out[:,19:] = torch.sigmoid(out[:,19:]) + torch.sqrt(torch.tensor(2.0,dtype=torch.float32))
        #print(out-target)
        if target is None:
            return out
        loss = self.compute_loss(out, target)
        if self.training:
            return loss
        return out, loss
       

    def _init_w(self):
        #init
         for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
         


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        if stride!=1:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride=(1,stride)), norm_layer(planes))
        else:
            downsample = None
        layers = []
        layers.append(block(self.inplanes, planes,stride=(1,stride),downsample=downsample))
        self.inplanes = planes
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)



def getM(dropout=None, weight=(1.0,1.0,1.0,100.0)):
    return Model(block=BasicBlock,dropout=dropout, weight=weight)
if __name__=="__main__":
    model = getM()
    print(model)