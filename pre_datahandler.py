import cv2
from PIL import Image
import numpy as np
import os
import time
from tqdm import tqdm
import shutil

def get_data(path = None,
             savedir = None):


    paths = [f'train_{i}_additional_to38cloud'for i in ['blue','green','red']]

    if os.path.exists(os.path.join(path,savedir)):
        shutil.rmtree(os.path.join(path,savedir))
    os.mkdir(os.path.join(path,savedir))

    pathb = os.listdir(os.path.join(path,paths[0]))
    pathg = os.listdir(os.path.join(path,paths[1]))
    pathr = os.listdir(os.path.join(path,paths[2]))


    #assert  (len(pathb) == len(pathg)) == len(pathr),"长度不一致"
    cur = 1
    for b,g,r in tqdm(zip(pathb,pathg,pathr),total = len(pathb),desc=f'{cur}/{len(pathb)}',postfix=dict,colour='#851022'):


        rindex = b.index('_')
        name = 'gt' + b[rindex:]

        name = name.split('.')[0]
        #print(name)

        b = os.path.join(path,paths[0],b)
        bimg = np.array(Image.open(b))
        g = os.path.join(path, paths[1], g)
        gimg = np.array(Image.open(g))
        r = os.path.join(path, paths[2], r)
        rimg = np.array(Image.open(r))


        #print(b,g,r)

        # la = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        # time.sleep(0.5)

        im0 = cv2.merge([bimg,gimg,rimg])



        #cv2.imshow('im0',im0)

        cv2.imwrite(os.path.join(path, savedir, name + '.tif'), im0)

        #cv2.waitKey(0)
        cur += 1



# get_data(path=r'D:\JinKuang\DomainAdaptiveDatasets\archive\95-cloud_training_only_additional_to38-cloud',
#          savedir = 'images')



# path1 = r'D:\JinKuang\DomainAdaptiveDatasets\archive\95-cloud_training_only_additional_to38-cloud\train_blue_additional_to38cloud\blue_patch_171_8_by_10_LC08_L1TP_028010_20160611_20170324_01_T1.TIF'
# b = np.array(Image.open(path1))
#
# path2 = r'D:\JinKuang\DomainAdaptiveDatasets\archive\95-cloud_training_only_additional_to38-cloud\train_green_additional_to38cloud\green_patch_171_8_by_10_LC08_L1TP_028010_20160611_20170324_01_T1.TIF'
# g = np.array(Image.open(path2))
#
#
# path3 = r'D:\JinKuang\DomainAdaptiveDatasets\archive\95-cloud_training_only_additional_to38-cloud\train_red_additional_to38cloud\red_patch_171_8_by_10_LC08_L1TP_028010_20160611_20170324_01_T1.TIF'
# r = np.array(Image.open(path3))
#
#
# im0 = cv2.merge([b,g,r])
#
#
# cv2.imshow('im0',im0)
# cv2.waitKey(0)


from glob import glob
from os import path as osp
from PIL import Image
import numpy as np
class Dataset_classifier(object):
    """
    统计数据集统计5个阶段的云层数据
    1. 含云量[0]
    2. 含云量(0,50]
    3. 含云量(50,90]
    4. 含云量(90,100)
    5. 含云量[100]
    """
    def __init__(self,
                 path = None,
                 suffix = '.jpg',
                 dataset = " "):
       self.images = glob(osp.join(path,'*'+suffix))

       self.zero = 0
       self.ten = 0
       self.trith = 0
       self.th = 0
       self.thirth = 0
       self.firth_ninth = 0
       self.ninth_hund = 0
       self.hund = 0
       assert dataset !=' ',"请输入数据集!!!"
       self.dataset = dataset
       self.logger = open(dataset+'.txt','w+')

    def forward(self):
        for image in self.images:
            # input： mask
            mask = Image.open(image).convert('L')
            w,h = mask.size

            area = w * h
            mask= np.array(mask)

            mask_sum = np.sum(mask)

            ratio = float(mask_sum / area)

            if ratio == 0.:
                self.zero += 1
            elif ((ratio > 0) & (ratio <= 0.10)):
                self.ten += 1
            elif ((ratio > 0.1) & (ratio <= 0.20)):
                self.trith += 1
            elif ((ratio > 0.2) & (ratio <= 0.30)):
                self.th += 1
            elif ((ratio > 0.3) & (ratio <= 0.50)):
                self.thirth += 1
            elif ((ratio > 0.5) & (ratio <= 0.90)):
                self.firth_ninth += 1
            elif ((ratio > 0.9) & (ratio < 1.)):
                self.ninth_hund += 1
            else:
                self.hund += 1
        self.result = {
            '[0]':self.zero,
            '(0,10]':self.ten,
            "(10,20]":self.trith,
            '(20,30]':self.th,
            '(30,50]':self.thirth,
            '(50,90]':self.firth_ninth,
            '(90,100)':self.ninth_hund,
            '[100]':self.hund
        }

    def write(self):
        self.logger.write(self.dataset + '\n')

        for data,value in self.result.items():
           
            self.logger.write(f'{data}:{value}'+ '\n')

if __name__ == '__main__':

    path = 'D:\JinKuang\TDA\Jin_Ytz\labels'
    suffix = '.png'

    dataset ='Yangtuze'

    cls = Dataset_classifier(path,
                             suffix,
                             dataset)
                             
    cls.forward()
    cls.write()
    