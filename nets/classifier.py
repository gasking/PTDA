import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.backbone import CBA

class FCclassifier(nn.Module):
    def __init__(self,
                 inc = None):
        super(FCclassifier, self).__init__()

        self.layer1 = nn.Sequential(
                      CBA(c = inc,
                          outc = inc //2,
                          k = 3,
                          p = 1),
                      CBA(c = inc // 2,
                          outc = inc //2 ,
                          k = 1,
                          p = 0),
                      CBA(c = inc //2 ,
                        outc = inc,
                        k = 3,
                        p = 1),
                      CBA(c = inc,
                        outc = inc,
                        k = 1,
                        p = 0),
        )

    def forward(self,x):
        idienity = x

        x = self.layer1(x)

        x = torch.add(idienity,x)

        x = F.normalize(x,dim = 1)

        return x



