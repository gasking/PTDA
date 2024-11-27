import torch
import torch.nn as nn
# -------------------------------------#
import numpy as np
import torch.optim as optim
from utils.utils_fit import epoch_fit
from utils.logger import Logger
import random
import shutil, os
import platform

# ---------------------------------------------------------------#
#                       导入解释器
# ---------------------------------------------------------------#
import argparse

from utils.config import Config
from utils.tools import warm_up, weight__init, fix_seed
from utils.callback import History
from torch.utils.data import DistributedSampler
from datasets.dataloader import UnetDataset, collate_seg
from torch.utils.data import DataLoader
from utils.summary import summary

# -------------------------------------#
# 更换模型
# -------------------------------------#
# 特征提取 + 像素判别器
from nets.deeplabv2 import get_deeplab_v2
# 像素领域判别器
from nets.discriminator import FCDiscriminator
# 评估函数
from utils.utils_eval import Eval

if __name__ == '__main__':

    parse_augment = argparse.ArgumentParser(description = 'UDA')

    # add baic augment

    parse_augment.add_argument('--init_epoch', type = int,
                               default = Config[ 'init_epoch' ], help = 'Max epoch...')

    parse_augment.add_argument('--freeze_epoch', type = int,
                               default = Config[ 'freeze_epoch' ], help = 'Max epoch...')

    parse_augment.add_argument('--unfreeze_epoch', type = int,
                               default = Config[ 'unfreeze_epoch' ], help = 'Max epoch...')

    # batch size
    parse_augment.add_argument('--batch_size', '--bs', type = int,
                               default = Config[ 'bs' ], help = 'train batch size')

    parse_augment.add_argument('--warm_up', type = int, default = 15, help = 'warm_up paramters...')

    # 训练集
    parse_augment.add_argument('--source_train_txt', '--st', type = str, default = Config[ 'source_train_txt' ],
                               help = 'source domain train file')

    parse_augment.add_argument('--target_train_txt', '-tt', type = str, default = Config[ 'target_train_txt' ],
                               help = 'target domain train file')

    # 测试集
    parse_augment.add_argument('--source_val_txt', '--sv', type = str, default = Config[ 'source_val_txt' ],
                               help = 'source domain val file')

    parse_augment.add_argument('--target_val_txt', '-tv', type = str, default = Config[ 'target_val_txt' ],
                               help = 'target domain val file')

    parse_augment.add_argument('--input_shape', '--size', type = list, default = [ 256, 256 ],
                               help = 'train image size')

    parse_augment.add_argument('--model_path', '--m', type = str, default = '',
                               help = 'pretrained model...')

    # 训练模式
    parse_augment.add_argument('--mode', '--md', type = str, default = 'co-train',
                               help = 'select train mode')

    # DDP train
    parse_augment.add_argument('--distributed','--dist',action = 'store_true',
                               default = False,help = 'DDP training...')

    # num_worker
    parse_augment.add_argument('--num_worker', '--nw', type = int, default = Config[ 'num_worker' ],
                               help = 'num_worker...')

    # decay rate
    parse_augment.add_argument('--decay_rate', '--d', type = float, default = Config[ 'decay_rate' ])

    # optimizer type
    parse_augment.add_argument('--optimizer_type', '--op', type = str, default = Config[ 'optimizer' ])

    # min lr
    parse_augment.add_argument('--min_lr', '--mlr', type = float, default = Config[ 'min_lr' ])

    # adam max lr
    parse_augment.add_argument('--adam_max_lr', '--adam_lr', type = float, default = Config[ 'adam_max_lr' ])

    # sgd max lr
    parse_augment.add_argument('--sgd_max_lr', '--sgd_lr', type = float, default = Config[ 'sgd_max_lr' ])

    # seed
    parse_augment.add_argument('--seed', '--s', type = int, default = 1234)

    # source domain
    parse_augment.add_argument('--source_domain', '--sd', type = str, required = True,
                               help = 'source domain name')

    # target domain
    parse_augment.add_argument('--target_domain', '--td', type = str, required = True,
                               help = 'target domain name')

    # save step
    parse_augment.add_argument('--save_step', '--save', type = int, default = Config[ 'save_step' ],
                               help = 'save model step')

    # accumulate
    parse_augment.add_argument('--accumulate', '--ac', type = int,
                               default = 4, help = 'accumulate grad...')

    # 解释
    args = parse_augment.parse_args()

    # -------------------------------------------------------------------------------#
    #                              多卡训练
    # -------------------------------------------------------------------------------#
    platformer = platform.system()

    if args.distributed:
        backendsname = 'gloo'if platformer == 'Windows' else 'nccl'

        #初始化后端
        torch.distributed.init_process_group(backend = backendsname)
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0 #设置



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

    # 测试集
    source_val_txt = args.source_val_txt
    target_val_txt = args.target_val_txt

    class_num = len(Config[ '_Class' ]) + 1

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

    # Evaluate
    eval = Eval(class_num = class_num)

    # ---------------------------------#
    #            权重保存路径
    # ---------------------------------#
    save_path = f'Pth\step1_{args.source_domain}_{args.target_domain}'

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    #设置多卡
    device = torch.device('cuda',local_rank) if torch.cuda.is_available() else torch.device('cpu')

    # -------------------------------------------#
    #                初始化模型
    # -------------------------------------------#
    featuremodel = get_deeplab_v2(num_classes = class_num, multi_level = True, backbone = 'resnet50')
    domainclassifer = FCDiscriminator(class_num)  # 辅助分支

    domainclassifer_main = FCDiscriminator(class_num)  # 主分支

    # -------------------------------------------#
    # 初始化日志器
    # -------------------------------------------#
    logger = Logger(file_path = os.path.join(save_path, 'looger'))

    model_path = args.model_path

    if os.path.exists(model_path) & (model_path != ''):
        miss = [ ]  # 缺少模块
        exist = [ ]  # 存在模块
        net = featuremodel.state_dict()  # 当前模型的图结构

        pretrainnet = torch.load(model_path, device)  # 预训练结构
    
        pretrainnet = pretrainnet[ 'featuremodel' ]

        

        temp = {}  # 收集共有层
        for k, v in pretrainnet.items():
            # filter 
            k = k[7:]

            if (k in net.keys()) & (np.shape(net[ k ]) == np.shape(v)):
                temp[ k ] = v
                exist.append(k)
            else:
                miss.append(k)
        net.update(temp)
        # 导入
        featuremodel.load_state_dict(net)

    if not os.path.exists(model_path):
        weight__init(featuremodel)  # 初始化网络参数

        weight__init(domainclassifer)
        logger.info('没有导入预训练权重，初始化网络参数')

    # --------------------------------------------#
    # 优化器
    # --------------------------------------------#

    # featureoptimizer = {
    #      'adam': optim.Adam(featuremodel.parameters(),lr = max_lr,weight_decay = Config['decay_rate']),
    #      'sgd': optim.SGD(featuremodel.parameters(),lr = max_lr,momentum = Config['momentum'],weight_decay = Config['decay_rate'] )
    # }[optimizer_type]

    featureoptimizer = optim.SGD(featuremodel.optim_parameters(2.5e-4), lr = 2.5e-4, momentum = 0.9,
                                 weight_decay = 0.0005)

    # domainclassiferoptimizer = {
    #    'adam': optim.Adam(domainclassifer.parameters(), lr = max_lr, weight_decay = Config[ 'decay_rate' ]),
    #    'sgd': optim.SGD(domainclassifer.parameters(), lr = max_lr, momentum = Config[ 'momentum' ],
    #                     weight_decay = Config[ 'decay_rate' ])
    # }[ optimizer_type ]

    domainclassiferoptimizer_aux = optim.Adam(domainclassifer.parameters(), lr = 1e-4, betas = (0.9, 0.99))

    domainclassiferoptimizer_main = optim.Adam(domainclassifer_main.parameters(), lr = 1e-4, betas = (0.9, 0.99))

    # --------------------------------------------#
    # 数据加载
    # --------------------------------------------#
    trainseg = UnetDataset(input_shape = input_shape, source_file = source_train_txt,
                           target_file = target_train_txt, num_classes = class_num,
                           mode = args.mode)

    trainSegData = DataLoader(trainseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                              shuffle = True,pin_memory = True)

    # 测试集
    valseg = UnetDataset(input_shape = input_shape, source_file = source_val_txt,
                         target_file = target_val_txt, num_classes = class_num,
                         mode = 'rank', train_val = True)
    valSegData = DataLoader(valseg, batch_size = freeze_batch, num_workers = num_worker, collate_fn = collate_seg,
                            shuffle = False)

    # wait for all porcesses to synchronize
    if args.distributed:
        torch.distributed.barrier()





    # 训练
    for epoch in range(init_epoch, unfreeze_epoch):

        if args.distributed:
            trainSegData.batch_sampler.sampler.set_epoch(epoch)

        # -----------------------------------#
        # 使用余弦退火法
        # -----------------------------------#
        warm_up([ featureoptimizer, domainclassiferoptimizer_aux, domainclassiferoptimizer_main ], freeze_epoch, epoch,
                min_lr, max_lr, warm_iter)

        epoch_fit(epoch, unfreeze_epoch, save_step,
                  [ featuremodel, domainclassifer, domainclassifer_main ],
                  [ featureoptimizer, domainclassiferoptimizer_aux, domainclassiferoptimizer_main ], trainSegData,
                  valSegData, device, logger, args, save_path, eval,local_rank)




