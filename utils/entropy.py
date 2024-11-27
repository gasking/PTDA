import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import cv2
import sys


path = os.getcwd()

sys.path.insert(0,path)


from datasets.dataloader import UnetDataset
from nets.deeplabv2 import get_deeplab_v2
from utils.tools import prob2entropy
from torch.utils.data import DataLoader



import argparse
from utils.config import Config
from utils.utils_eval import Eval


from threading import Thread


class entrophyrank(object):
    def __init__(self,ratio = 0.5):

        super(entrophyrank, self).__init__()

        self.ratio = ratio #将样本划分为简单样本和复杂样本

        self.eval = Eval(2)

    def update(self,lists,args):
    #----------------------------#
    #     创建保存推理标签文件夹
    #---------------------------#



        if os.path.exists(args.save_path):
            import shutil
            shutil.rmtree(args.save_path)

        os.mkdir(args.save_path)

        # 熵值保存路径
        save_entropy_path = args.entropy_save_path

        if os.path.exists(save_entropy_path):
            import shutil
            shutil.rmtree(save_entropy_path)

        os.mkdir(save_entropy_path)

        copy_list = copy.deepcopy(list)

        line = sorted(lists,key = lambda l:l[-1])
        l = len(line)
        ml = int(self.ratio * l) + 1

        easy_example = line[:ml]

        hard_example = line[ml:]

        thres = 0.5


        #-----------------------------------------------------------#
        #                        Inferce
        #-----------------------------------------------------------#

        with open('easy_example.txt','w+') as f:
            for con in easy_example:


                    #------------------------------------------------------------#
                    #                     熵值可视化
                    #------------------------------------------------------------#
                    entropy = con[-2][0].detach().cpu().numpy()
                    #index = np.array((entropy < thres ),dtype=np.float32)
                    #entropy = entropy * index
                    #entropy = np.argmax(entropy,axis=0)

                    entropy_background = np.array(entropy[0, ...] * 255, dtype=np.uint8)
                    entropy_foreground = np.array(entropy[1, ...] * 255, dtype=np.uint8)

                    heatmap = (0.5 * entropy_background + 0.5 * entropy_foreground)

                    heatmap = np.array(heatmap, np.uint8)
                    entropy = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    #
                    # print(entropy.shape)


                    # 获取目标域简单样本进行训练并将预测结果作为标签进行训练
                    image = con[0]

                    imagename = con[1]
                    name = con[1][con[1].rindex('\\') + 1:].split('.')[0]
                    #
                    label = torch.argmax(image,dim = 1)[0]
                    label = label.detach().cpu().numpy()

                    # gt
                    gt = con[2][0].numpy()
                    self.eval.init(gt,label)

                # Miou = self.eval.get_Miou()
                #
                # if Miou[1] > 0:

                    save = os.path.join(path,args.save_path)

                    cv2.imwrite(f'{save}/{name}_low.png',label*255)

                    cv2.imwrite(f'{save_entropy_path}/{name}_entropy_low.png',entropy )



                    cv2.imwrite(f'{save}/{name}_gt.png',gt*255)



                    #print(gt.shape,label.shape)
                    self.eval.init(gt,label)

                    #mask = image.replace('.jpg', '.png').replace('test', 'testlab')
                    # 写文件
                    f.write(imagename +' '+f'{save}\\{name}_low.png''\n')

        with open('hard_example.txt', 'w+') as f:
            for con in  hard_example:
                image = con[0]

                entropy = con[-2][0].detach().cpu().numpy()
                # index = np.array((entropy < thres ),dtype=np.float32)
                # entropy = entropy * index
                # entropy = np.argmax(entropy,axis=0)

                entropy_background = np.array(entropy[0, ...] * 255, dtype=np.uint8)
                entropy_foreground = np.array(entropy[1, ...] * 255, dtype=np.uint8)


                heatmap = (0.5 * entropy_background + 0.5 * entropy_foreground)

                heatmap = np.array(heatmap,np.uint8)
                entropy = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


                imagename = con[1]
                name = con[1][con[1].rindex('\\') + 1:].split('.')[0]
                #
                label = torch.argmax(image, dim=1)[0]
                label = label.detach().cpu().numpy()

                save = os.path.join(path, args.save_path)

                # gt
                gt = con[2][0].numpy()

                self.eval.init(gt,label)

                # Miou = self.eval.get_Miou()
                # if Miou[1] > 0.45:

                cv2.imwrite(f'{save}\\{name}_gt.png', gt *255)
                    #
                cv2.imwrite(f'{save}\\{name}_high.png', label*255)

                cv2.imwrite(f'{save_entropy_path}/{name}_entropy_high.png', entropy )

                #self.eval.init(gt, label)
                    # mask = image.replace('.jpg', '.png').replace('test', 'testlab')
                # 写文件
                f.write(imagename + ' ' + f'{save}\\{name}_high.png''\n')

        print(f'{"*" * 20}样本划分完毕{"*" * 20}\t')

        self.eval.show()

        return copy_list


if __name__ == '__main__':
    #----------------------------------------#
    #                 参数配置
    #----------------------------------------#
    parse = argparse.ArgumentParser(description = 'Entropy rank')

    parse.add_argument('--ratio','--r',type = float,
                       default = 0.7,help = 'split ratio')

    parse.add_argument('--model','--m',type = str,
                       default=r'D:\JinKuang\UDS\step1_WHU_Clouds26\5.pth',help = 'Inference Model')
    parse.add_argument('--target_train_txt','--tt',type = str,
                       default = r'D:\\JinKuang\\GASKING_Expiment_DA\\source_train.txt',help = 'target domain train txt')
    #-------------------------------------------------------------------------------------------------------------------#
    #                                               可视化伪标签
    #-------------------------------------------------------------------------------------------------------------------#
    parse.add_argument('--save_path','--save',type = str,default = 'pseudo_label',
                       help = 'pseudo_label save path')

    # -------------------------------------------------------------------------------------------------------------------#
    #                                               可视化熵值
    # -------------------------------------------------------------------------------------------------------------------#
    parse.add_argument('--entropy_save_path', '--entropy_save_path', type=str, default='entropy_label',
                       help='entropy_label save path')

    args = parse.parse_args()



    # 熵值
    rank = entrophyrank(ratio = args.ratio)


    #-----------------------------------------------------------#
    #                      模型导入
    #-----------------------------------------------------------#
    model_path = args.model
    device = torch.device('cuda')

    model = torch.load(model_path,map_location = device)
    feature = get_deeplab_v2(num_classes=2,multi_level=True,backbone='resnet50')

    from torch.nn import DataParallel
    #feature = DataParallel(feature)
    feature.load_state_dict(model['featuremodel'])

    feature = feature.to(device)



    feature.eval()


    target_data = UnetDataset(source_file = args.target_train_txt,
                              target_file = args.target_train_txt,
                              mode='rank',input_shape=(256,256))

    traindataloader = DataLoader(target_data,
                                 batch_size = 1,
                                 num_workers = 4,
                                 shuffle = False
                                 )
    interp = nn.Upsample(size=(256, 256), mode='bilinear',
                         align_corners=True)
    lists = [] #收集
    for ind,batch in enumerate(traindataloader):
        with torch.no_grad():
            _, _, _, targetimage, targetlabel, targetpng, targetname = batch

            targetimage = targetimage.to(device)


            pred_aux,pred = feature(targetimage)
            pred = interp(pred)



            #------------------------------------------------------------------------#
            #                               熵值
            #------------------------------------------------------------------------#
            targetmap = prob2entropy(F.softmax(pred))
            lists.append([F.softmax(pred),targetname[0],targetpng,targetmap,targetmap.sum().data.item()])

    # task = Thread(target = rank.update,args=(lists,args))
    #
    # task.start()

    rank.update(lists,args)
