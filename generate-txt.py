from glob import glob
import os
import random

# -------------------------------------------------------#
#  生成训练文件,文件以此格式存放
#  VOCdevkit
#    VOC2023
#      Annotations #标签文件
#      JPEGImages  #原始图像
#      MASKImages  #掩码图像

def Generate(JPG = None,
             MASK = None,
             type = None
             ):

    TrainTxt = type + '_train.txt'
    TestTxt = type + '_val.txt'

    precent = 0.9
    datafiles = set(map(lambda t: t.split('\\')[ -1 ].split('.')[ 0 ], glob(JPG + '\*.tif')))
    sample = int(len(datafiles) * precent)
    traindata = set(random.sample(datafiles, sample))
    testdata = datafiles - traindata

    traindata, testdata = list(traindata), list(testdata)
    trainfile = open(TrainTxt, 'w+')
    for id in traindata:
        trainfile.write(os.path.join(JPG, "%s.tif" % id) + ' ')
        trainfile.write(os.path.join(MASK, "%s.tif" % id) + '\n')

    valfile = open(TestTxt, 'w+')
    for id in testdata:
        valfile.write(os.path.join(JPG, "%s.tif" % id) + ' ')
        valfile.write(os.path.join(MASK, "%s.tif" % id) + '\n')
    trainfile.close()
    valfile.close()


class DataHandler(object):
    def __init__(self,
                 path = None,
                 train_ratio = 0.7,
                 val_ratio = 0.1,
                 task = ''):

        self.path = path

        self.file = (os.listdir(path))

        trainval = int((train_ratio + val_ratio) * len(self.file))

        self.trainval = random.sample(self.file,trainval)

        tl = int(((train_ratio / (train_ratio + val_ratio)) * trainval))
        self.train = random.sample(self.trainval,tl)


        # 打开保存文件进行保存
        self.tr = open(f'{task}_train.txt','w+')
        self.vl = open(f'{task}_val.txt','w+')
        self.ts = open(f'{task}_test.txt','w+')

    def __call__(self, *args, **kwargs):

        for name in self.file:

            if name in self.trainval:
                if name in self.train:
                 name = name.split('.')[0]
                 self.tr.write(name+'\n')
                else:
                    name = name.split('.')[0]
                    self.vl.write(name+'\n')
            else:
                name = name.split('.')[0]
                self.ts.write(name+'\n')












if __name__ == "__main__":
    random.seed(0) #制定随机数

    path = r'D:\JinKuang\TDA\HRC_WHU-copy\images'
    D = DataHandler(path = path,task='WHU')
    D()

    path = r'D:\JinKuang\TDA\Jin_Ytz\images'
    D1 = DataHandler(path = path,task='Clouds26')
    D1()
    # 源域
    # Generate(JPG = r'D:\JinKuang\Cloud\HRC_WHU-copy\images',
    #          MASK = r'D:\JinKuang\Cloud\HRC_WHU-copy\labels',
    #          type = 'source')
    # 目标域
    # Generate(JPG = r'D:\JinKuang\Cloud\95-cloud\images',
    #          MASK = r'D:\JinKuang\Cloud\95-cloud\labels',
    #          type = 'target')






