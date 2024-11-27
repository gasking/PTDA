from nets.deeplabv2 import get_deeplab_v2
#------------------------------------------#
#               特征提取器
#------------------------------------------#
import cv2
import os
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F



class Grad_CAM(object):
    def __init__(self, modules = None,
                 pth_path = None,
                 num_classes = 2):

        assert pth_path != None, '请输入权重...'
        

        
        pass
