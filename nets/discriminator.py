import torch
import torch.nn as nn
from nets.backbone import CBA

class FCDiscriminator(nn.Module):
    def __init__(self,
                 num_classes = None, #64
                 ndf = 64):

        super(FCDiscriminator, self).__init__()
        self.layer = nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )


    def forward(self,x):


        return self.layer(x)

#---------------------------------------------#
#               Downsample
#---------------------------------------------#
if __name__ == "__main__":
    x = torch.randn((1,64,256,256))

    model = FCDiscriminator(inc = 64)

    output = model(x)

    output.backward(output.data)

    print(output.shape)