#--------------------------------#
# 配置文件
#--------------------------------#

Config = {

    'num_worker':4,
    'decay_rate':1e-3, # if use Adam
    'adam_max_lr': 2.5e-4,
    'sgd_max_lr':2.5e-4,
    'min_lr': 1e-6,
    'momentum':0.,
    'optimizer':'adam',

    'init_epoch':0,
    'freeze_epoch': 15,
    'unfreeze_epoch':50,

    # train 15 50
    # uda 5 15

    'bs':16,

    'mode':'co-train',
    #'mode':'rank',



    '_Class':('cloud',), #使用元组为一个类别时，记得加上逗号

    'pretrained': True,
    'model_path':'model_data/18.pth',
    'save_step':1,
    'input_shape':(256,256),



    'source_train_txt': r'D:\JinKuang\GASKING_Expiment_DA\source_train.txt',
    'target_train_txt': r'D:\JinKuang\GASKING_Expiment_DA\target_train.txt',

    'source_val_txt': r'D:\JinKuang\GASKING_Expiment_DA\source_val.txt',
    'target_val_txt': r'D:\JinKuang\GASKING_Expiment_DA\target_val.txt',

    # 'source_val_txt': r'D:\JinKuang\GASKING_Expiment_DA\target_val.txt',
    # 'target_val_txt': r'D:\JinKuang\GASKING_Expiment_DA\95_target_val.txt',

    'easy_split': r'D:\JinKuang\GASKING_Expiment_DA\source_train.txt',
    'hard_split': r'D:\JinKuang\GASKING_Expiment_DA\target_train.txt',


    'mean':[ 0.485, 0.456, 0.406 ],
     'std':[ 0.229, 0.224, 0.225 ],

    'platte':[[191,246,195],[200,210,240]]  #二分类
}