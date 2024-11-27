import torch
import torch.nn as nn
from nets.backbone import DeConv


class UPSampler(nn.Module):
    def __init__(self,
                 inc = None,
                 num_classes = 1+1):
        super(UPSampler, self).__init__()

        self.layer1 = DeConv(in_channel = inc,
                             out_channel = inc*2)

        self.layer2 = DeConv(in_channel = inc*2,
                             out_channel = inc*4)

        self.layer3 = DeConv(in_channel = inc*4,
                             out_channel = num_classes)
        #self.fin = nn.Conv2d(in_channels=inc,out_channels=num_classes,kernel_size=1)

    def forward(self,x):

        x = self.layer1(x)
        x = self.layer2(x)

        return (self.layer3(x))



if __name__ == '__main__':
    x = torch.empty((1,32,64,64))


    model = UPSampler(inc = 32,
                      num_classes = 2)

    output = model(x)


    output.backward(output.data)


    print(output.shape)