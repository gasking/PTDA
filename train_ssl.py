import torch
import torch.nn as nn
from utils.config import Config
from utils.tools import warm_up, weight__init, fix_seed
from utils.callback import History
# from datasets.dataloader_uda import UnetDataset, collate_seg

from datasets.dataloader import UnetDataset, collate_seg
from datasets.dataloader import UnetDataset as UDADataSet
from datasets.dataloader import collate_seg as UDAcollate_seg
from torch.utils.data import DataLoader
from utils.summary import summary
from nets.deeplabv2 import get_deeplab_v2
from nets.discriminator import FCDiscriminator
# -------------------------------------#


import numpy as np
import torch.optim as optim
from utils.utils_ssl_old import epoch_fit
#from utils.utils_asl import epoch_fit

from utils.logger import Logger
import random, math
from copy import deepcopy
import shutil, os
from utils.tools import ModelEMA
from utils.utils_eval import Eval
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':

    parse_augment = argparse.ArgumentParser(description='UDA')

    # add baic augment

    parse_augment.add_argument('--init_epoch', type=int,
                               default=Config['init_epoch'], help='Max epoch...')

    parse_augment.add_argument('--freeze_epoch', type=int,
                               default=Config['freeze_epoch'], help='Max epoch...')

    parse_augment.add_argument('--unfreeze_epoch', type=int,
                               default=Config['unfreeze_epoch'], help='Max epoch...')

    # batch size
    parse_augment.add_argument('--batch_size', '--bs', type=int,
                               default=Config['bs'], help='train batch size')

    parse_augment.add_argument('--warm_up', type=int, default=0, help='warm_up paramters...')

    parse_augment.add_argument('--source_train_txt', '--st', type=str, default=Config['source_train_txt'],
                               help='source domain train file')

    parse_augment.add_argument('--target_train_txt', '-tt', type=str, default=Config['target_train_txt'],
                               help='target domain train file')

    parse_augment.add_argument('--source_val_txt', '--sv', type=str, default=Config['source_val_txt'],
                               help='source domain val file')

    parse_augment.add_argument('--target_val_txt', '-tv', type=str, default=Config['target_val_txt'],
                               help='target domain val file')

    parse_augment.add_argument('--input_shape', '--size', type=list, default=[256, 256],
                               help='train image size')

    parse_augment.add_argument('--model_path', '--m', type=str, default='',
                               help='pretrained model...')

    # 训练模式
    parse_augment.add_argument('--mode', '--md', type=str, default='co-train',
                               help='select train mode')

    # num_worker
    parse_augment.add_argument('--num_worker', '--nw', type=int, default=Config['num_worker'],
                               help='num_worker...')

    # decay rate
    parse_augment.add_argument('--decay_rate', '--d', type=float, default=Config['decay_rate'])

    # optimizer type
    parse_augment.add_argument('--optimizer_type', '--op', type=str, default=Config['optimizer'])

    # min lr
    parse_augment.add_argument('--min_lr', '--mlr', type=float, default=Config['min_lr'])

    # adam max lr
    parse_augment.add_argument('--adam_max_lr', '--adam_lr', type=float, default=Config['adam_max_lr'])

    # sgd max lr
    parse_augment.add_argument('--sgd_max_lr', '--sgd_lr', type=float, default=Config['sgd_max_lr'])

    # seed
    parse_augment.add_argument('--seed', '--s', type=int, default=1234)

    # source domain
    parse_augment.add_argument('--source_domain', '--sd', type=str, required=True,
                               help='source domain name')

    # target domain
    parse_augment.add_argument('--target_domain', '--td', type=str, required=True,
                               help='target domain name')

    # save step
    parse_augment.add_argument('--save_step', '--save', type=int, default=Config['save_step'],
                               help='save model step')

    # accumulate
    parse_augment.add_argument('--accumulate', '--ac', type=int,
                               default=4, help='accumulate grad...')
    

    # Ablation study

    # intra-domain adaptation
    parse_augment.add_argument('--intra',action = 'store_true',help = 'is using intra-domain adaptation...')

    # anchor
    parse_augment.add_argument('--anchor',action = 'store_true',help = 'is using anchor Network...')

    # Pixcontrast loss
    parse_augment.add_argument('--pixcontrast',action = 'store_true',help = 'is using pixcontrast loss...')

    # consist loss
    parse_augment.add_argument('--consist',action = 'store_true',help = 'is using consist loss...')

    #关键参数消融实验
    parse_augment.add_argument('--seg_para',type = float,default = 2,required = True,help = 'Segloss weigt...')
    parse_augment.add_argument('--contrast_para',type =float,default = 0.001,required = True,help = 'contrast weught')



    # 解释器
    args = parse_augment.parse_args()

    # ------------------------------------------------------------------------------#
    #                               参数赋值
    # ------------------------------------------------------------------------------#
    init_epoch = args.init_epoch
    freeze_epoch = args.freeze_epoch
    unfreeze_epoch = args.unfreeze_epoch

    num_worker = args.num_worker

    freeze_batch = args.batch_size

    input_shape = args.input_shape

    # ---------------------------------------------------#
    #                 训练标签
    # ---------------------------------------------------#
    source_train_txt = args.source_train_txt
    target_train_txt = args.target_train_txt

    # -----------------------------------------------------#
    #                    测试集
    # -----------------------------------------------------#
    source_val_txt = args.source_val_txt
    target_val_txt = args.target_val_txt

    class_num = len(Config['_Class']) + 1

    # -----------------------------------------------------#
    #                    模型评估
    # -----------------------------------------------------#
    eval = Eval(class_num)

    model_path = args.model_path

    save_step = args.save_step

    min_lr = args.min_lr

    warm_iter = args.warm_up

    # ------------------------------#
    # 优化器种类
    # ------------------------------#
    optimizer_type = args.optimizer_type

    max_lr = args.adam_max_lr if optimizer_type == 'adam' else args.sgd_max_lr

    # fix seed
    seed = args.seed

    fix_seed(seed)

    # ---------------------------------#
    #            权重保存路径
    # ---------------------------------#
    save_path = f'Pth\step3_{args.source_domain}_{args.target_domain}'

    if not os.path.exists(save_path):
        
     os.mkdir(save_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # -------------------------------------------#
    # 初始化模型
    # -------------------------------------------#
    featuremodel = get_deeplab_v2(num_classes=class_num,
                                  multi_level=True,
                                  backbone='resnet50')

    # -----------------------------------------------#
    #                 辅助领域判别分支
    # -----------------------------------------------#
    domainclassifer_aux = FCDiscriminator(class_num)
    # 主领域判别分支
    domainclassifer_main = FCDiscriminator(class_num)

    # -------------------------------------------#
    # 初始化日志器
    # -------------------------------------------#
    logger = Logger(file_path=os.path.join(save_path, 'logger'))

    # -------------------------------------------#
    # 模型结构初始化
    # -------------------------------------------#

    model_path = args.model_path

    # 导入模型分割图信息
    if (model_path != ''):
        miss = []  # 缺少模块
        exist = []  # 存在模块
        net = featuremodel.state_dict()  # 当前模型的图结构

        domainnet_aux = domainclassifer_aux.state_dict()


        domainnet_main = domainclassifer_main.state_dict()

        pretrainnet = torch.load(model_path, device)  # 预训练结构

        # pretraindomainnet_aux = pretrainnet['domainmodel_aux']
        # pretraindomainnet_main = pretrainnet['domainmodel_main']

        pretrainnet = pretrainnet['featuremodel']

        # -----------------------------------------------------#
        #                   TODO trans to
        # -----------------------------------------------------#
        temp = {}  # 收集共有层
        for k, v in pretrainnet.items():
            #k = k[7:]
            if (k in net.keys()) & (np.shape(net[k]) == np.shape(v)):
                temp[k] = v
                exist.append(k)
            else:
                miss.append(k)
        net.update(temp)
        # 导入
        featuremodel.load_state_dict(net)

        # ----------------------------------------------#
        #               辅助分支领域判别
        # ----------------------------------------------#
        # temp = {}  # 收集共有层
        # for k, v in pretraindomainnet_aux.items():
        #     if (k in domainnet_aux.keys()) & (np.shape(domainnet_aux[k]) == np.shape(v)):
        #         temp[k] = v
        #         exist.append(k)
        #     else:
        #         miss.append(k)
        # domainnet_aux.update(temp)
        # # 导入
        # domainclassifer_aux.load_state_dict(domainnet_aux)

        # ----------------------------------------------#
        #               主分支领域判别
        # ----------------------------------------------#
        # temp = {}  # 收集共有层
        # for k, v in pretraindomainnet_main.items():
        #     if (k in domainnet_main.keys()) & (np.shape(domainnet_main[k]) == np.shape(v)):
        #         temp[k] = v
        #         exist.append(k)
        #     else:
        #         miss.append(k)
        # domainnet_main.update(temp)
        # # 导入
        # domainclassifer_main.load_state_dict(domainnet_main)

    # --------------------------------------------------------------#
    #                        初始化学生模型
    #                  采用学生教室模型的方式进行自监督训练
    #                 这一部分不进行梯度求导，只用来进行推理
    # --------------------------------------------------------------#
    Student_featuremodel = ModelEMA(featuremodel)

    # ---------------------------------------------------------------#
    #     不进行参数更新，STOP GRADIENT
    # ---------------------------------------------------------------#
    old_model = get_deeplab_v2(num_classes=class_num,
                               multi_level=True,
                               backbone='resnet50')

    if (model_path != ''):
        miss = []  # 缺少模块
        exist = []  # 存在模块
        net = old_model.state_dict()  # 当前模型的图结构
        pretrainnet = torch.load(model_path, device)  # 预训练结构
        pretrainnet = pretrainnet['featuremodel']
        temp = {}  # 收集共有层
        for k, v in pretrainnet.items():
            #k = k[7:]
            if (k in net.keys()) & (np.shape(net[k]) == np.shape(v)):
                temp[k] = v
                exist.append(k)
            else:
                miss.append(k)
        net.update(temp)
        # 导入
        old_model.load_state_dict(net)

    # --------------------------------------------#
    # 优化器
    # --------------------------------------------#

    featureoptimizer = optim.SGD(featuremodel.optim_parameters(2.5e-4), lr=2.5e-4, momentum=0.9, weight_decay=0.0005)

    # domainclassiferoptimizer = {
    #    'adam': optim.Adam(domainclassifer.parameters(), lr = max_lr, weight_decay = Config[ 'decay_rate' ]),
    #    'sgd': optim.SGD(domainclassifer.parameters(), lr = max_lr, momentum = Config[ 'momentum' ],
    #                     weight_decay = Config[ 'decay_rate' ])
    # }[ optimizer_type ]

    domainclassiferoptimizer_aux = optim.Adam(domainclassifer_aux.parameters(), lr=1e-4, betas=(0.9, 0.99))

    domainclassiferoptimizer_main = optim.Adam(domainclassifer_main.parameters(), lr=1e-4, betas=(0.9, 0.99))

    # featureoptimizer = {
    #     'adam': optim.Adam(featuremodel.optim_parameters(max_lr), lr=max_lr, weight_decay=Config['decay_rate']),
    #     'sgd': optim.SGD(featuremodel.optim_parameters(max_lr), lr=max_lr, momentum=Config['momentum'],
    #                      weight_decay=Config['decay_rate'])
    # }[optimizer_type]
    #
    # domainclassiferoptimizer_aux = {
    #     'adam': optim.Adam(domainclassifer_aux.parameters(), lr=max_lr, weight_decay=Config['decay_rate']),
    #     'sgd': optim.SGD(domainclassifer_aux.parameters(), lr=max_lr, momentum=Config['momentum'],
    #                      weight_decay=Config['decay_rate'])
    # }[optimizer_type]
    #
    # domainclassiferoptimizer_main = {
    #     'adam': optim.Adam(domainclassifer_main.parameters(), lr=max_lr, weight_decay=Config['decay_rate']),
    #     'sgd': optim.SGD(domainclassifer_main.parameters(), lr=max_lr, momentum=Config['momentum'],
    #                      weight_decay=Config['decay_rate'])
    # }[optimizer_type]

    # --------------------------------------------#
    # 训练数据加载
    # --------------------------------------------#
    trainseg = UnetDataset(input_shape=input_shape, source_file=source_train_txt,
                           target_file=target_train_txt, num_classes=class_num,
                           mode=args.mode,augment=True)
    trainSegData = DataLoader(trainseg, batch_size=freeze_batch, num_workers=num_worker, collate_fn=collate_seg,
                              shuffle=True)

    # 测试集
    valseg = UDADataSet(input_shape=input_shape, source_file=source_val_txt,
                         target_file=target_val_txt, num_classes=class_num,
                         mode='rank',train_val=True)
    valSegData = DataLoader(valseg, batch_size=freeze_batch, num_workers=num_worker, collate_fn=UDAcollate_seg,
                            shuffle=False,pin_memory=True)

    old_model = old_model.eval()

    # 训练文件
    for epoch in range(init_epoch, unfreeze_epoch):
        # -----------------------------------#
        # 使用余弦退火法
        # -----------------------------------#
        warm_up([featureoptimizer, domainclassifer_aux, domainclassifer_main], freeze_epoch, epoch, min_lr, max_lr,
                warm_iter)

        epoch_fit(epoch, freeze_epoch, unfreeze_epoch, save_step,
                  [featuremodel, domainclassifer_aux, domainclassifer_main],
                  [featureoptimizer, domainclassiferoptimizer_aux, domainclassiferoptimizer_main], trainSegData,
                  valSegData, device, logger, Student_featuremodel, old_model, args, save_path, eval)