from __future__ import print_function

import os.path as osp
import copy
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np
import .vit

class Setr(nn.Module):

    def __init__(self, num_class, dim, depth, heads, batch_size, trans_img_size, bilinear = False):
        super(Setr, self).__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.batch_size = batch_size
        self.trans_img_size = trans_img_size
        #Transformer unit (encoder)
        self.transformer = vit.ViT(
            image_size = trans_img_size,
            patch_size = 16,
            num_classes = 64, #not used
            dim = dim,
            depth = depth,    #number of encoders
            heads = heads,    #number of heads in self attention
            mlp_dim = 3072,   #hidden dimension in feedforward layer
            channels = 3,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.channel_reduction = nn.Conv2d(in_channels=dim, out_channels=1024, kernel_size=3, padding=1)
        self.n_class = num_class
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn3     = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.bn4     = nn.BatchNorm2d(64)
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.up_final = nn.Upsample(scale_factor=2, mode='bilinear')   	
        else:
            self.up = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
            self.up_final = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.classifier = nn.Conv2d(64, num_class, kernel_size=1)

    def forward(self, x):   
        bs = x.size(0)     
        score = self.transformer(x)
        score = torch.reshape(score, (bs,self.dim, int(self.trans_img_size/16), int(self.trans_img_size/16)))

        score = self.up(self.bn1(self.relu(self.deconv1(score))))    
        score = self.up(self.bn2(self.relu(self.deconv2(score)))) 
        score = self.up(self.bn3(self.relu(self.deconv3(score))))
        score = self.up_final(self.bn4(self.relu(self.deconv4(score))))  
        score = self.classifier(score)                   

        return score  # size=(N, n_class, x.H/1, x.W/1)   
