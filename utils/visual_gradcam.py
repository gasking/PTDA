import os
import sys
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# 获取当前工作空间
path = os.getcwd()
sys.path.insert(0,path)

from nets.criterion import Dice_loss
from nets.deeplabv2 import get_deeplab_v2
from utils.tools import resize,_norm



class Grad_CAM(object):
    def __init__(self, modules = None,
                 pth_path = None,
                 input_shape = (256,256)):

        assert pth_path != None, '请输入权重...'

        pass
