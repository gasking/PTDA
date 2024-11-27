#------------------------------------#
# 绘制损失图
# 平滑曲线
#------------------------------------#
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.config import Config

class History():
    def __init__(self,trainfile,valfile,model,log = 'logs',weight = 0.85,device = None):
        self.max_cost = float('inf')

        self.tf = trainfile
        self.vf = valfile

        self.trainloss = open(trainfile,'w+')
        self.valloss = open(valfile,'w+')

        self.log = log
        self.tl = [] #记录数据长度
        self.vl = []
        #-----------------------------#
        # 平滑曲线
        #-----------------------------#
        self.smooth_tl = []
        self.smooth_vl = []

        self.weight = weight

        #------------------------------------------#
        # tensorbaord 可视化
        #------------------------------------------#
        self.writer = SummaryWriter(self.log)

        input = torch.randn((1,3)+Config['input_shape']).to(device)
        try:
         self.writer.add_graph(model,input)
        except:
         assert "无法写入"
    def _compare(self,a):
        self.max_cost = min(self.max_cost,a)
        if self.max_cost == a: #Min
           return True
        return False
    # ----------------------------------#
    # 记录训练过程的损失
    # ----------------------------------#
    def record_loss(self,epoch,trainlossdata,vallossdata):
        self.trainloss.write(str(trainlossdata) + '\n')
        self.valloss.write(str(vallossdata) + '\n')
        self.append_loss(epoch,trainlossdata,vallossdata)

    def get_loss(self):
        self.trainloss.close()
        self.valloss.close()

        self.tl = [float(v.strip()) for v in open(self.tf,'r',encoding = 'utf-8').readlines()]
        self.vl = [ float(v.strip()) for v in open(self.vf,'r',encoding = 'utf-8').readlines() ]

        self._smooth(self.tl, self.smooth_tl)
        self._smooth(self.vl, self.smooth_vl)

        self._plot()

    def append_loss(self,epoch,trainloss,valloss):
        #self.tl.append(trainloss)

        #self.vl.append(valloss)

        self.writer.add_scalar('train loss',trainloss,epoch)
        self.writer.add_scalar('val loss', valloss, epoch)

        #self._smooth(self.tl,self.smooth_tl)
        #self._smooth(self.vl,self.smooth_vl)

        #self._plot()

    def _smooth(self,a,b):
        last = a[ 0 ]
        for cur in a:
         v = last * self.weight + (1. - self.weight) * cur
         last = v
         b.append(v)

    def _plot(self):
        plt.figure()


        plt.plot(range(len(self.tl)),self.tl,'red',linewidth = 2,label = 'train_loss')
        plt.plot(range(len(self.vl)),self.vl,'blue',linewidth = 2,label = 'val_loss')
        #--------------------------------#
        # 平滑曲线
        #--------------------------------#
        plt.plot(range(len(self.smooth_tl)),self.smooth_tl, 'green',linestyle = '--', linewidth = 2, label = 'smooth train_loss')
        plt.plot(range(len(self.smooth_vl)),self.smooth_vl, '#6DB9EF', linestyle = '--', linewidth = 2, label = 'smooth val_loss')


        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc = "upper right")

        plt.savefig(os.path.join(self.log, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

if __name__ == "__main__":
   history = History('../logs/train_loss.txt', '../logs/val_loss.txt')  # 正无穷
   history.get_loss()