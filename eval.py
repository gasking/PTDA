import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import time
from nets.unet import SegHead
from nets.Upsample import UPSampler
from nets.deeplabv2 import get_deeplab_v2
from utils.tools import prob2entropy
from torch.utils.data import DataLoader
from datasets.dataloader import UnetDataset
from utils.utils_eval import Eval
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 云的掩码
Cloud = np.array([
    [ 255 ,255,255],
    [ 50,120,90]
]) / 255.


def plot_(fig,
          data,
          label,
          name,
          mode = None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)



    data = (data - x_min) / (x_max - x_min) #normalization

    color = {
        'source':{1: '#E72929', 0: '#41C9E2'},
        'target':{1: '#5755FE', 0: '#FDA403'}
    }

    # color = {
    #     'source':  'r',
    #     'target':  'b'
    # }
    #map_color = {i: color[ i ] for i in range(2)}
    map_size = {i: 10 for i in range(2)}



    #color = list(map(lambda x: map_color[ x ], label))
    size = list(map(lambda x: map_size[ x ], label))






    # print(len(color),len(size),data.shape)
    #

    l = data.shape[0]
    # print('start draw...')
    # for ind in range(l):
    #     plt.scatter(data[ind,0],data[ind,1],c = color[mode][label[ind]])
    # 以颜色代表特征
    # print('start draw...')
    # plt.scatter(data[ :, 0 ], data[ :, 1 ])

    # 以数字代表特征

    print('start draw...')
    for ind in range(l):
        plt.text(data[ ind, 0 ], data[ ind, 1 ],str(int(label[ind])),color=color[mode][label[ind]],
				 fontdict={ 'size': 8})



    # plt.show()

    return fig


def draw(feature, label, name, mode = None,
         form_mode = None,fig = None,tsne = None):
    b,c,h,w = feature.shape

    feature = feature.permute((0,2,3,1))
    #label = label.permute((1, 2, 0))


    feature = feature.reshape((-1, c))
    label = label.reshape((-1,))



    TSNE_result = tsne.fit_transform(feature.data.cpu().numpy())

    #print(f'TSNE: {TSNE_result.shape}')
    TSNE_label = label.data.cpu().numpy()

    fig = plot_(fig,TSNE_result, TSNE_label, name,form_mode)
    return fig


if __name__ == '__main__':

    from  utils.config import Config
    # -----------------------------------------------------------#
    #                      模型导入
    #  r'D:\JinKuang\UDS\Pth\entropy\step3_Clouds26_WHU\last.pth'
    #  r'D:\JinKuang\UDS\Pth\entropy\step3_WHU_Clouds26\last.pth'
    # 
    # -----------------------------------------------------------#
    _1 = r'D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth'
    _2 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_intrada\last.pth'
    _3 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_intrada_anchor\last.pth'
    _4 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_intrada_anchor_pix\10.pth'
    _5 =  r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_intrada_anchor_pix_consist\last.pth' #r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_intrada_anchor_pix_consist\last.pth'

    model_path1 = r'D:\JinKuang\UDS\Pth\paper\entropy\step3_Clouds26_WHU\last.pth'#r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_intrada\last.pth'
    model_path2 = r'second_Clouds26_WHU\10.pth'
    model_path3 = r'threelogs\19.pth'

    #------------------------------------------------------------#
    #                       上采样
    #------------------------------------------------------------#
    interp = nn.Upsample(size=(Config['input_shape'][0], Config['input_shape'][1]), mode='bilinear',
                              align_corners=True)

    model_path4 = r'D:\JinKuang\UDS\Pth\paper\entropy\step1_Clouds26_WHU\20.pth'

    model_path5 = r'D:\JinKuang\UDS\Pth\step3_WHU_95Cloud\5.pth'
    is_show = True
    types = ['Clouds26toWHU','WHUtoClouds26','WHUto95']

    
    mode = types[0]


    device = torch.device('cuda')

    #-----------------------------------------------------------------------#
    #                          关键参数的消融实验
    #-----------------------------------------------------------------------#
    key0 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_Key_0\last.pth'
    key1 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_Key_1\last.pth'
    key2 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_Key_2\last.pth'
    key3 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_Key_3\last.pth'
    key4 = r'D:\JinKuang\UDS\Pth\step3_Clouds26_WHU_Key_4\last.pth'

    model = torch.load(_5, map_location = device)

    feature = get_deeplab_v2(num_classes=2,multi_level=True,backbone='resnet50')

    from  torch.nn import DataParallel

    #feature = DataParallel(feature)

    feature.load_state_dict(model[ 'featuremodel' ])

    feature = feature.to(device)


    feature.eval()


    evaltor = Eval(2)

    #source_file =  r'D:\JinKuang\GASKING_Expiment_DA\source_val.txt'
    #target_file =  r'D:\JinKuang\GASKING_Expiment_DA\target_val.txt'

    source_file = r'D:\\JinKuang\\GASKING_Expiment_DA\\tsne_Clouds26_WHU.txt'
    target_file =r'D:\\JinKuang\\GASKING_Expiment_DA\\tsne_Clouds26_WHU.txt'

    target_data = UnetDataset(source_file = source_file,
                              target_file = target_file,
                              input_shape = Config['input_shape'],
                              mode = 'rank',
                              train_val=True, #不进行黑边填充推理
                              augment=False
                              )


    traindataloader = DataLoader(target_data,
                                 batch_size =8,
                                 num_workers = 4,
                                 shuffle = False
                                 )
    lists = [ ]  # 收集
    print(len(traindataloader))

    savedir = rf'Vis_{mode}'

    if os.path.exists(savedir):
        import shutil

        shutil.rmtree(savedir)
    os.mkdir(savedir)

    print("start T_SNE...")

    tsne = TSNE(n_components=2,  random_state=0, perplexity=5,early_exaggeration=35,learning_rate='auto')
    for ind, batch in enumerate(traindataloader):
        with torch.no_grad():
            sourceimage, sourcelabel, sourcepng, targetimage, _, targetpng, targetname = batch

            targetimage = targetimage.to(device)
            sourceimage = sourceimage.to(device)

            targetpng = targetpng.to(device)
            sourcepng = sourcepng.to(device)

            pred_seg, pred = feature(targetimage)



            spred_seg,spred = feature(sourceimage)

            if not is_show:
                pred_seg = interp(pred_seg)
                pred = interp(pred)
                spred_seg = interp(spred_seg)
                spred = interp(spred)



            targetOneHotlabel = F.softmax(pred,dim=1)


            #targetOneHotlabel = torch.argmax((targetOneHotlabel), dim=1)  # 目标域


            aux_target = torch.argmax(pred_seg,dim=1)
            main_target = torch.argmax(pred,dim=1)
            targetOneHotlabel = (aux_target  | main_target) 



            sourceOneHotlabel =F.softmax(spred,dim=1)
            sourceOneHotlabel = torch.argmax((sourceOneHotlabel), dim=1)  # 源域





            tpred = pred_seg.clone()
            spred = spred.clone()

            #SourceOneHotlabel = F.interpolate(sourcepng.unsqueeze(dim=1), size=(h, w), mode='nearest').squeeze(dim=1)
            save = targetname[0][targetname[0].rindex('\\') + 1:]

            save = save.split('.')[0]
            if is_show:

             fig = plt.figure()

             ax = plt.gca()

             #draw(pred, OneHotlabel.long(), name = targetname[0],mode= mode)  # 可视化
             # tpred = torch.cat((tpred,spred),dim=1)
             # targetOneHotlabel = torch.cat((targetOneHotlabel,sourceOneHotlabel),dim=1)
             fig = draw(tpred,targetOneHotlabel, name=targetname[0], mode=mode,form_mode='target',fig=fig,tsne = tsne)  # 可视化
             draw(spred, sourceOneHotlabel, name=targetname[0], mode=mode, form_mode='source',fig=fig,tsne = tsne)

             # plt.xticks([-0.5,0.5])
             #
             # plt.yticks([-0.5,0.5])
             #
             # plt.title(targetname[0])

             #save = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + f'_{mode}.tif'

             ax.spines['right'].set_visible(False)
             ax.spines['left'].set_visible(False)
             ax.spines['top'].set_visible(False)
             ax.spines['bottom'].set_visible(False)

             plt.xticks([])
             plt.yticks([])

             save = targetname[0][targetname[0].rindex('\\') + 1:]

             save = save.split('.')[0]

             plt.savefig(os.path.join(savedir,f'{save}_{mode}.tif'), dpi=150, bbox_inches='tight')

             print("Generate TSNE finined...")

            else:
                targetmap = targetOneHotlabel[0].cpu().numpy()

                targetpng = targetpng.cpu().numpy()

                #print(targetmap.shape,targetpng.shape)



                evaltor.init(targetmap, targetpng)


                image = targetimage[0].detach().cpu().numpy().transpose((1,2,0))
                evaltor.vis(image,targetmap,save,savedir)

    evaltor.show()







