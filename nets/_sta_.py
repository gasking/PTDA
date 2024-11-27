import torch
import torch.nn as nn

class UP(nn.Module):
    def __init__(self,inc,outc):
        super(UP, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.conv1 = nn.Conv2d(in_channels = inc + outc,out_channels = outc,kernel_size = 3,padding = 1)
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


class Down(nn.Module):
    def __init__(self,inc,outc):
        super(Down, self).__init__()
        filter = [inc,outc,outc]
        self.down = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels = filter[i],
                          out_channels = filter[i + 1], kernel_size = 3,padding = 1),
                nn.BatchNorm2d(filter[i + 1]),
                nn.ReLU(inplace = True)
            ) for i in range(2)
        )
        self.maxpool = nn.MaxPool2d(stride = 2,kernel_size = 3,padding = 1)


    def forward(self,x):
        x = self.maxpool(x)
        for layer in self.down:
            x = layer(x)

        return x


class Unet(nn.Module):
    def __init__(self,num_classes = 1 + 1):
        super(Unet, self).__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels = 3,
                          out_channels = 64, kernel_size = 3,padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace = True)
            )
        self.layer2 = Down(64,128)
        self.layer3 = Down(128,256)
        self.layer4 = Down(256,512)
        self.layer5 = Down(512,1024)


        self.up1 = UP(1024,512)
        self.up2 = UP(512,256)
        self.up3 = UP(256,128)
        self.up4 = UP(128,64)


        self.out = nn.Conv2d(64,num_classes,kernel_size = 1,padding = 0,stride = 1)

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

        out = self.out(up4)
        return out



if __name__ == '__main__':
    x = torch.randn((1,3,512,512))
    net = Unet(2)
    out = net(x)
    print(out.shape)