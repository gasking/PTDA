import torch
import torch.nn as nn
import torch.nn.functional as F



#----------------------------------#
# dice loss
#----------------------------------#
def Dice_loss(pre,tar):
    assert pre.shape == tar.shape,"Compute dice loss error!"

    pre = F.softmax(pre,dim = 1)

    union = (pre * tar).sum()

    A1 = pre.sum()
    A2 = tar.sum()

    loss = union/(A1 + A2 - union + 1e-5)
    return 1 - loss.mean()


#----------------------------------#
# focal loss
#----------------------------------#
def focal_loss(pre,tar,alpha = 0.25,bate = 2.):

    assert pre.shape == tar.shape, "Compute focal loss error!"

    b,c,h,w = pre.shape
    if c <3:
        pre = F.sigmoid(pre)
    else:
        pre = F.softmax(pre)

    obj = (tar != 0 ).float()
    noobj = (tar == 0).float()

    obj_loss = (-1) * alpha * (1 - pre)**bate*torch.log(pre) * obj

    noobj_loss = (-1) * (1 - alpha)*pre**bate*torch.log(1 - pre) * noobj

    loss = obj_loss + noobj_loss

    return loss

#-------------------------------------------#
# score
#-------------------------------------------#
def score(pre,tar,thes = 0.5):
    assert pre.shape == tar.shape,"Compute dice loss error!"
    pre = F.softmax(pre,dim = 1)
    #------------------------------#
    #
    #------------------------------#
    pre = torch.ge(pre,thes)

    union = (pre * tar).sum()

    A1 = pre.sum()
    A2 = tar.sum()

    cost = union/(A1 + A2 - union + 1e-5)
    return cost.mean()


class PixContrasrt(nn.Module):
    def __init__(self,
                 thresh = 5,
                 max_epoch = None):
        super(PixContrasrt, self).__init__()

        self.dist = nn.Parameter(torch.tensor(0.2,
                                              requires_grad=True))
        self.thresh = thresh

        self.max_epoch = max_epoch

    def forward(self, feature, label,iteration):

        pass
