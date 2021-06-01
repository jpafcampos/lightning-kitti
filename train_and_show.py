import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from models.setr.setr import Setr
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import wandb

from helper import *

import random

DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)


class SegModel(pl.LightningModule):
    def __init__(self):
        super(SegModel, self).__init__()
        self.batch_size = 4
        self.learning_rate = 1e-3
#         self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
#         self.net = UNet(num_classes = 19, bilinear = False)
#         self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
        self.net = Setr(num_class=19, dim=1024, depth=3, heads=3, 
                        batch_size=self.batch_size, trans_img_size=512, bilinear = False)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
        ])
        self.trainset = semantic_dataset(split = 'train', transform = self.transform)
        self.testset = semantic_dataset(split = 'test', transform = self.transform)
        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_nb) :
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)
        loss_val = F.cross_entropy(out, mask, ignore_index = 250)
#         print(loss.shape)
        return {'loss' : loss_val}
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size = 1, shuffle = True)


model = SegModel()
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath = './',
    save_top_k = [1],
    verbose = True, 
    monitor = 'loss',
    mode = 'min'
)
trainer = pl.Trainer(gpus = 1, max_nb_epochs = 30, checkpoint_callback = checkpoint_callback, early_stop_callback = None)
trainer.fit(model)


model = SegModel()
checkpoint = torch.load('./_ckpt_epoch_70.ckpt', map_location = lambda storage, loc : storage)
model.load_state_dict(checkpoint['state_dict'])


model.cuda()
model.net.eval()
testloader = model.test_dataloader()
img = next(iter(testloader))
img = img.float().cuda()
y = model.forward(img)
mask_pred = y.cpu().detach().numpy()
mask_pred_bw = np.argmax(mask_pred[0], axis = 0)

unorm = UnNormalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
img2 = unorm(img)
img2 = img.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()

fig, axes = plt.subplots(2, 1)
axes[0].imshow(img2[0])
axes[1].imshow(mask_pred_bw)
plt.savefig('output.png')
plt.show()