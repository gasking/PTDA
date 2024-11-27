import torch
import numpy as np
import cv2
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image  # 这个读取数据是RGB
from torch.utils.data import DataLoader, Dataset
import random
from utils.tools import resize
from utils.config import Config
from datasets.transformer import *



class UnetDataset(Dataset):
    def __init__(self, input_shape = (512, 512),
                 source_file = None,
                 target_file = None,
                 num_classes = 1 + 1,
                 mode = 'rank'):

        self.image_size = input_shape

        self.mode = mode

        self.num_class = num_classes #类别

        self.source = [] #收集原始图像
        self.target= [] #收集掩码图像
        sourcelines = open(source_file)
        targetlines = open(target_file)


        self.data_aug = Data_Augment([#RandomHue(),
                      # RandomSaturation(),
                      # RandomBrightness(),
                     # RandomHFlip(),
                      #RandomVFlip(),
                      #RandomBlur(),
                     # RandomRotate(),
                      Noramlize()        ])

        for line in sourcelines.readlines():
            splited = line.strip().split(' ')[0]
            self.source.append(splited)

        for line in targetlines.readlines():
            splited = line.strip().split(' ')[0]
            self.target.append(splited)


        sourcelines.close()
        targetlines.close()
        self.l = max(len(self.source),len(self.target))

    def handler(self,fname):

        image = Image.open(fname).convert('RGB')

        w, h = image.size

        scalew, scaleh = self.image_size[ 0 ] / w, self.image_size[ 1 ] / h
        scale = min(scaleh, scalew)
        neww, newh = int(scale * w), int(scale * h)

        dx = self.image_size[ 0 ] - neww
        dy = self.image_size[ 1 ] - newh
        new_image = Image.new("RGB", self.image_size, (0, 0, 0))
        image = image.resize((neww, newh))



        if random.uniform(0, 1) > 0.5:
            dx //= 2
            dy //= 2

        new_image.paste(image,(dx,dy))
        image = np.array(new_image, dtype = np.float32)
        image = np.transpose(image,(2,0,1))

        image = self._norm(image)



        return image

    def __getitem__(self, idx):
        ind1 = idx % len(self.source)
        sourcename = self.source[ind1]


        ind2 = idx % len(self.target)
        targetname = self.target[ind2]


        sourceimage = self.handler(sourcename )

        targetimage = self.handler(targetname)


        return sourceimage,targetimage


    def __len__(self):
        return self.l
    # ------------数据增强-----------------#
    def _norm(self, img):
        img = img/255.
        # img -= self.mean
        # img /= self.std
        return img




def collate_seg(batch):



     sourceimage, targetimage =  [],[]

     for sim,tim in batch:
        sourceimage.append(sim)
        targetimage.append(tim)


     sourceimage = np.array(sourceimage)
     sourceimage = torch.from_numpy(sourceimage).float()

     targetimage = np.array(targetimage)
     targetimage = torch.from_numpy(targetimage).float()


     return sourceimage,targetimage




if __name__ == '__main__':
 def gt():
    data = UnetDataset(source_file = '../utils/easy_example.txt',
                       target_file = '../utils/hard_example.txt')
    train_loader = DataLoader(data, batch_size = 8, shuffle = True,collate_fn = collate_seg)
    #train_iter = iter(train_loader)
    for i,batch in enumerate(train_loader):
        sourceimage,targetimage = batch

        for ind in range(sourceimage.shape[0]):

         sim = sourceimage[ind].numpy()
         sim = np.transpose(sim,(1,2,0)) * 255.
         sim = sim.astype(np.uint8)[...,::-1]



         tim = targetimage[ ind ].numpy()
         tim = np.transpose(tim, (1, 2, 0)) * 255.
         tim = tim.astype(np.uint8)[...,::-1]

         cv2.imshow('sim',sim)
         cv2.imshow('tim',tim)
         cv2.waitKey(0)


 gt()


