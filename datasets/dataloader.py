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
                 mode = 'co-train',
                 augment = False,
                 train_val = False):

        self.image_size = input_shape

        self.mode = mode

        self.num_class = num_classes #类别

        self.source = [] #收集原始图像
        self.target= [] #收集掩码图像
        sourcelines = open(source_file)
        targetlines = open(target_file)

        if augment: #推理
            self.data_aug = Data_Augment([RandomHue(),
                          #RandomSaturation(),
                          #RandomBrightness(),
                          RandomHFlip(),
                          RandomVFlip(),
                         #  RandomBlur(),
                         # RandomRotate(),
                          Noramlize()        ])

        else:
            self.data_aug = Data_Augment([

                Noramlize()])

        for line in sourcelines.readlines():
            splited = line.strip().split()
            self.source.append([splited[ 0 ],splited[1]])

        for line in targetlines.readlines():
            splited = line.strip().split()
            self.target.append([splited[ 0 ],splited[1]])


        sourcelines.close()
        targetlines.close()

        self.train_val = train_val

        if not self.train_val:
         self.l = max(len(self.source),len(self.target))
        else:
            self.l = len(self.target)

    def handler(self,fname,maskname):
        image = Image.open(fname).convert('RGB')

        mask = Image.open(maskname).convert('L')

        assert np.array(image.size).all() == np.array(mask.size).all(), "Not Match! Error"


        if not self.train_val:
            w, h = image.size

            scalew, scaleh = self.image_size[ 0 ] / w, self.image_size[ 1 ] / h
            scale = min(scaleh, scalew)
            neww, newh = int(scale * w), int(scale * h)

            dx = self.image_size[ 0 ] - neww
            dy = self.image_size[ 1 ] - newh

            # 128 过于接近云
            new_image = Image.new("RGB", self.image_size, (0, 0, 0))
            image = image.resize((neww, newh))

            new_mask = Image.new("L", self.image_size, (0))
            mask = mask.resize((neww, newh))

            if random.uniform(0, 1) > 0.5:
                dx //= 2
                dy //= 2

            new_image.paste(image, (dx, dy))
            new_mask.paste(mask, (dx, dy))
        else :
            new_image = image.resize(self.image_size)
            new_mask = mask.resize(self.image_size)



        image = np.array(new_image, dtype = np.float32)

        # t = image.copy()
        # t = t.astype(np.uint8)
        # crop = image[dy:dy+newh,dx:dx+neww,:]
        # crop = crop.astype(np.uint8)
        # #crop = cv2.resize(crop,(ow,oh))
        # print(crop.shape)
        # cv2.imshow('im', t)
        # cv2.imshow('crop',crop)
        # cv2.waitKey(0)

        # ----------------------------------#
        # 二分类
        # ----------------------------------#
        mask = np.array(new_mask)  # 单通道图像

        result = self.data_aug({
            'image': image,
            'mask': mask
        })

        image = result[ 'image' ]
        mask = result[ 'mask' ]

        modify_png = np.zeros_like(mask)
        modify_png[ mask > 0 ] = 1

        # -------------------------------#
        # 多分类
        # -------------------------------#
        # for c in range(self.num_class):
        #     mask[mask==c] = c

        T_mask = np.zeros((self.image_size[ 1 ], self.image_size[ 0 ], self.num_class))

        # --------------------------------------#
        # 两种构建one-hot编码形式
        # --------------------------------------#
        for c in range(self.num_class):
            T_mask[ modify_png == c, c ] = 1
        T_mask = np.transpose(T_mask, (2, 0, 1))
        # T_mask = np.eye(self.num_class)[mask.reshape(-1)] #
        # T_mask = np.reshape(T_mask,(self.image_size[1],self.image_size[0],self.num_class))

        # vision
        """
        back = T_mask[0,...]
        fg = T_mask[1,...]
        cv2.imshow('im',image.astype(np.uint8))
        cv2.imshow('bg',back)
        cv2.imshow('fg',fg)
        cv2.waitKey(0)
        """

        img = np.transpose(image, (2, 0, 1))

        return img,T_mask,modify_png

    def __getitem__(self, idx):
        ind1 = idx % len(self.source)
        sourcename = self.source[ind1][0]
        sourcemask = self.source[ind1][1]

        ind2 = idx % len(self.target)
        targetname = self.target[ind2][0]
        targetmask = self.target[ind2][1]

        sourceimage,sourcelabel,sourcepng = self.handler(sourcename,
                                                         sourcemask)

        targetimage,targetlabel,targetpng = self.handler(targetname,
                                                         targetmask)

        if self.mode == 'co-train':
         return sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng
        elif self.mode == 'rank':
         return sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng,targetname
        else:
            raise NotImplementedError("没有该训练模式")

    def __len__(self):
        #return self.l
        return len(self.target)
    # ------------数据增强-----------------#
    def _norm(self, img):
        img = img/255.
        # img -= self.mean
        # img /= self.std
        return img


def convert(image,label,png):
    image = np.array(image)
    image = torch.from_numpy(image).float()

    label = torch.from_numpy(np.array(label)).float()

    png = np.array(png)

    png = torch.from_numpy(png).long()

    return image,label,png

def collate_seg(batch):


    #mode = Config['mode']

    l = len(batch[0])


    if l == 6:
     sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng= [],[],[],\
                                                                          [],[],[]

     for sim,slabel,spng,tim,tlabel,tpng in batch:
        sourceimage.append(sim)
        sourcelabel.append(slabel)
        sourcepng.append(spng)

        targetimage.append(tim)
        targetlabel.append(tlabel)
        targetpng.append(tpng)


     sourceimage,sourcelabel,sourcepng = convert(sourceimage,
                                                sourcelabel,
                                                sourcepng)

     targetimage,targetlabel,targetpng = convert(targetimage,
                                                targetlabel,
                                                targetpng)

     return  sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng
    elif l == 7:
        sourceimage, sourcelabel, sourcepng, targetimage, targetlabel, targetpng ,targetname= [ ], [ ], [ ], \
                                                                                   [ ], [ ], [ ],[ ]

        for sim, slabel, spng, tim, tlabel, tpng, name in batch:
            sourceimage.append(sim)
            sourcelabel.append(slabel)
            sourcepng.append(spng)

            targetimage.append(tim)
            targetlabel.append(tlabel)
            targetpng.append(tpng)
            targetname.append(name)

        sourceimage, sourcelabel, sourcepng = convert(sourceimage,
                                                      sourcelabel,
                                                      sourcepng)

        targetimage, targetlabel, targetpng = convert(targetimage,
                                                      targetlabel,
                                                      targetpng)

        return sourceimage, sourcelabel, sourcepng, targetimage, targetlabel, targetpng,targetname

    else:
        raise NotImplementedError('没有该模式')


if __name__ == '__main__':
 def gt():
    data = UnetDataset(source_file = Config['source_train_txt'],
                       target_file = Config['target_train_txt'])
    train_loader = DataLoader(data, batch_size = 8, shuffle = True,collate_fn = collate_seg)
    #train_iter = iter(train_loader)
    for i,batch in enumerate(train_loader):

        print(len(batch),type(batch))
        sourceimage,sourcelabel,sourcepng,targetimage,targetlabel,targetpng,targetname = batch

        for ind in range(sourceimage.shape[0]):
         print(targetname[ind])
         sim = sourceimage[ind].numpy()
         sim = np.transpose(sim,(1,2,0)) * 255.
         sim = sim.astype(np.uint8)[...,::-1]

         for c in range(sourcelabel.shape[1]):
             T = sourcelabel[ind].numpy()

             T = np.transpose(T,(1,2,0))

             cv2.imshow(f"im_{c}",T[...,c])

         tim = targetimage[ ind ].numpy()
         tim = np.transpose(tim, (1, 2, 0)) * 255.
         tim = tim.astype(np.uint8)[...,::-1]

         cv2.imshow('sim',sim)
         cv2.imshow('tim',tim)
         cv2.waitKey(0)


 gt()


