import torch
import torch.nn as nn
from nets.backbone import resnet50,DeConv,CBA



class UP(nn.Module):
    def __init__(self,inc,outc):
        super(UP, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels = inc,out_channels = outc,kernel_size = 3,padding = 1)
        self.relu1 = nn.ReLU(inplace = True)

        self.out = nn.Sequential(nn.Conv2d(in_channels = outc, out_channels = outc, kernel_size = 3, padding = 1),
                                 nn.ReLU(inplace = True))

    def forward(self,x,x1):
        #--------------------------#
        # x1降采样
        #--------------------------#
        outputs = torch.cat((x1,self.up(x)),dim = 1)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        return self.out(outputs)

class SegHead(nn.Module):
    def __init__(self,pretrained = False,num_class = 1 + 1):
        super(SegHead, self).__init__()
        self.backbone = resnet50(pretrained)
        for i in range(len(self.backbone)):
            setattr(self,f'layer{i + 1}',self.backbone[i])
        # 3072,1536,768,320
        # 2048 1024 512 256
        in_filter = [ 3072,1536,768,320]
        out_filter = [1024,512,256,64]

        for i in range(len(in_filter)):
            setattr(self,f'up{i + 1}',UP(in_filter[i],out_filter[i]))


        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor = 2),
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size = 3,padding = 1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64,num_class,kernel_size = 3,padding = 1,bias = False)

    def forward(self,x):
        #--------------------------------#
        # 降采样
        #--------------------------------#

        feat1 = self.layer1(x) #第一层
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)


        #--------------------------------#
        # 上采样
        #--------------------------------#

        up1 = self.up1(feat5,feat4)
        up2 = self.up2(up1,feat3)
        up3 = self.up3(up2,feat2)
        up4 = self.up4(up3,feat1)



        #--------------------------------#
        # output
        #--------------------------------#
        out = self.up_conv(up4)

        out = self.out(out)
        return out

    def freeze_backbone(self):
        for mm in self.backbone:
         for m in mm.parameters():

            m.requires_grad = False

    def unfreeze_backbone(self):
        for mm in self.backbone:
            for m in mm.parameters():
                m.requires_grad = True


if __name__ == '__main__':
    x = torch.randn((1,3,512,512))
    net = SegHead()
    out = net(x)
    print(out.shape)


