import torch
import torch.nn as nn
from nets.deeplabv2 import get_deeplab_v2


from utils.tools import resize,_norm
from utils.config import Config
import numpy as np
from PIL import Image
import cv2
from copy import deepcopy
import torch.nn.functional as F



class OD():
  def __init__(self,model_path,num_class):
      #------------------------------------------------#
      # 模型初始化
      #------------------------------------------------#
      self.num_class = num_class #类别
      self.input_shape = Config['input_shape']

      self.class_name = ('__background__','cloud')

      self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


      self.feature = get_deeplab_v2(num_classes = len(self.class_name),
                                    multi_level=True,
                                    backbone='resnet50')

      model = torch.load(model_path,map_location = self.device)
      model = model['featuremodel']

      self.interp = nn.Upsample(size = (Config['input_shape'][0],Config['input_shape'][1]),mode = 'bilinear',align_corners = True)





      self.feature.load_state_dict(model)




      self.feature = self.feature.eval()

      #------------------------------------------------#
      # 调色板
      #------------------------------------------------#
      self.platte =  [ (176, 235, 100), (180, 200, 220), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                            (128, 64, 12)]
      self.platte = np.array(self.platte,dtype = np.int32)


  def _resize(self,image,input_shape):
      # image is Image
      new_image,neww, newh ,dx,dy,w,h = resize(image,input_shape)


      img = np.array(new_image,dtype = np.float32)

      img = _norm(img)

      return new_image,img,neww, newh,dx,dy,w,h

  @torch.no_grad()
  def detect_img(self,image):
      #----------------------------#
      # 输入为PIL
      #----------------------------#

      image = image.convert('RGB')
      old_image = deepcopy(image)

      _image,image,newh,neww,dx,dy,ow,oh = self._resize(old_image,self.input_shape)

      # print(neww,newh)
      # cv2.imshow('im',image)
      # cv2.waitKey(0)

      img = np.transpose(image,(2,0,1))


      input = torch.from_numpy(img).float() #

      input = torch.unsqueeze(input,dim = 0)

      seg1,seg = self.feature(input)

      seg1 = self.interp(seg1)
      seg = self.interp(seg)



      #------------------------------------#
      # Get seg result
      #------------------------------------#
      # if self.num_class < 3:
      #     seg = torch.sigmoid(seg)
      # else:
      seg = F.softmax(seg1,dim = 1)

      seg = seg.cpu().data.numpy()[0] #convert numpy
      seg = np.transpose(seg,(1,2,0))


      #-------------------------------------#
      # original image
      #-------------------------------------#

      or_seg = seg[dy:dy+neww,dx:dx + newh]

      or_seg = cv2.resize(or_seg,(ow,oh),cv2.INTER_LINEAR)

      # cv2.imshow('1',or_seg[...,0]* np.array(old_image)[...,0])
      # cv2.imshow('2', or_seg[...,1] * np.array(old_image)[...,0])
      # cv2.waitKey(0)

      pr = np.argmax(or_seg,axis = -1) #对于类别维度进行取最大值

      print(np.reshape(np.unique(pr), [ -1 ]))

      newshape = pr.shape
      mask = np.zeros((int(newshape[0]),int(newshape[1]),3),dtype = np.float32)

      for c in range(self.num_class):
          mask[ ...,0] += (pr == c) * self.platte[ c ][ 0 ]
          mask[ ...,1] += (pr == c) * self.platte[c][1]
          mask[ ...,2] += (pr == c) * self.platte[c][2]
      print(np.reshape(np.unique(mask), [ -1 ]))
      mask = mask.astype(np.uint8)
      #mask = np.transpose(mask,(1,0,2))
      mask = Image.fromarray(mask)


      new_image = Image.blend(old_image,mask,0.7)


      im0 = np.array(old_image)[...,0]
      mask0 = np.array(mask)[...,0]//128

      im1 = im0 * mask0
      cv2.imshow('im0',np.array(new_image,dtype=np.uint8))
      cv2.imshow('mask',np.array(mask,dtype=np.uint8))
      cv2.imshow('im2',np.array(old_image,dtype=np.uint8))
      cv2.waitKey(0)


      #new_image.show()
      #new_image.save('save.png')






  def refine_image(self,image,dx,dy,ow,oh,neww,newh):
      img = image.crop((dx,dy,dx + neww,dy + newh))

      img = img.resize((ow,oh))

      return img


if __name__ == '__main__':
    #model_path = r'F:\JinKuang\USD\nets\weight.pth'
    mode = 0

    model_path1 = r'Pth\entropy\step3_Clouds26_WHU\last.pth'
    model_path2 = r'Protype\c_weight_20.pth'
    model_path3 = r'F:\JinKuang\2024_experiment_paper\USD\threelogs\last_epoch_15.pth'


#D:\JinKuang\TDA\HRC_WHU-copy\images\03_27_03_34_09_46.jpg
    model_path4 = r'D:\JinKuang\UDS\threelogs/19.pth'
    #model_path = r'logs/last_epoch_25.pth' if mode == 0 else r'F:\JinKuang\USD\nets\weight_10.pth'
    num_class = 1 + 1#D:\JinKuang\Cloud\CloudS26\images\0_8160_3360.jpg

    # """
    # F:\JinKuang\2024_experiment_paper\Jin_Ytz_WHU\USD\pselabel\2024_03_29_15_19_57_0.jpg
    # D:\JinKuang\Cloud\HRC_WHU-copy\images\snow_16_11.jpg
    # D:\JinKuang\Cloud\HRC_WHU-copy\images\vegetation_19_21.jpg
    # D:\JinKuang\Cloud\HRC_WHU-copy\images\vegetation_17_10.jpg

    # """
    od = OD(model_path1,num_class)
    while True:
        imagepath = input("请输入需要预测的图片路径：")


        try:
         image = Image.open(imagepath)

         png = imagepath.replace('images', 'labels').replace('jpg', 'png')

         # mask = cv2.imread(png)
         # cv2.imshow('mask', mask * 255)


        except:
         print("Path error! please again input!!!")
         continue
        else:
         od.detect_img(image)