import torch
import torch.nn as nn
import torch.nn.functional as F
#---------------------------------------------#
# cross anttention
#---------------------------------------------#
class CrossAttention(nn.Module):
    def __init__(self,
                 inc):
        super(CrossAttention, self).__init__()

        hidden = inc
        self.p = 1 / (inc ** 0.5)

        # 使用卷积变化代替全连接
        self.linear1 = nn.Linear(inc,hidden)

        self.linear2 = nn.Linear(inc,hidden)

        self.linear3 = nn.Linear(inc,hidden)

    def forward(self,query,key,value):
        #----------------------------#
        # 计算多模态相似度
        # query 来自分割
        # key value 来自扩散模型
        #----------------------------#
        b,c,h,w = query.shape

        query = query.contiguous().view((b,-1))
        key = key.contiguous().view((b, -1))
        value = value.contiguous().view((b, -1))

        #print(query.shape,key.shape,value.shape)

        query = self.linear1(query)
        key = self.linear2(key)
        value = self.linear3(value)


        xout = F.softmax((query @ key.T) / self.p,dim = -1)

        xout = xout @ value

        return xout


class MutilCrossAttention(nn.Module):
    def __init__(self,
                 heads,
                 inc):
        super(MutilCrossAttention, self).__init__()
        hidden = inc // heads

        self.heads = heads

        self.layer = nn.ModuleList(
            CrossAttention(hidden) for i in range(heads)
        )

    def forward(self,query,key,value):

        b,c,h,w = query.shape



        query = query.chunk(self.heads,dim = 1)
        key = key.chunk(self.heads, dim = 1)
        value = value.chunk(self.heads, dim = 1)

        features = []
        for i in range(len(query)):
            x = self.layer[i](query[i],key[i],value[i])

            features.append(x)

        features = torch.cat(features,dim = -1)

        return features.view((b,c,h,w))
