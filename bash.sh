#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Clouds -> WHU

#python train.py  --sd Clouds26 --td WHU --save 5 --unfreeze_epoch 25 \
#    --bs 32 --source_train_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_train.txt --target_train_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_train.txt \
#     --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt
#
#
## 熵值排序
#python utils/entropy.py --r 0.6 --model Pth/step1_Clouds26_WHU/last.pth --target D:\\JinKuang\\GASKING_Expiment_DA\\target_train.txt \



#python train_ssl.py --model_path Pth/step1_Clouds26_WHU/last.pth --sd Clouds26 --td WHU --save 5 --unfreeze 25 \
#      --bs 16 --source_train_txt easy_example.txt --target_train_txt hard_example.txt \
#     --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# WHU -> Clouds26

# python train.py   --sd WHU --td Clouds26 --save 1 --unfreeze_epoch 50 \
#     --bs 32 --source_train_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_train.txt --target_train_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_train.txt \
#      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt

# python utils/entropy.py --r 0.7 --model D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_WHU_Clouds26\\last.pth --target D:\\JinKuang\\GASKING_Expiment_DA\\source_train.txt \



# python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_WHU_Clouds26\\25.pth --sd WHU --td Clouds26 --save 1 --unfreeze 25 \
#       --bs 16   --consist --source_train_txt easy_example.txt --target_train_txt hard_example.txt \
#       --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#
#
#
#
#
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
## WHU -> 95Cloud
#
#python train.py  --sd WHU --td 95Cloud --save 5 --unfreeze_epoch 50 \
#     --bs 32 --source_train_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_train.txt --target_train_txt D:\\JinKuang\\GASKING_Expiment_DA\\95_target_train.txt \
#      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\95_target_val.txt
#
#
## 熵值排序
#python utils/entropy.py --r 0.6 --model Pth/step1_WHU_95Cloud/last.pth --target D:\\JinKuang\\GASKING_Expiment_DA\\95_target_train.txt \

#python utils/entropy.py --r 0.6 --model Pth/25.pth --target D:\\JinKuang\\GASKING_Expiment_DA\\95_target_train.txt \


#
#
#
#python train_ssl.py --model_path Pth/25.pth --sd WHU --td 95Cloud --save 5 --unfreeze 25  --num_worker 0 \
#       --bs 16 --source_train_txt hard_example.txt --target_train_txt easy_example.txt \
#       --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\95_target_val.txt
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#




#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#   消融实验
#   intraDA
# python utils/entropy.py --r 0.6 --model D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth --target D:\\JinKuang\\GASKING_Expiment_DA\\target_train.txt \

# python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth --sd Clouds26 --td WHU_intrada --save 1 --unfreeze 25  \
#       --bs 16 --source_train_txt easy_example.txt --target_train_txt hard_example.txt \
#       --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt

#   intraDA + anchor
# python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth --sd Clouds26 --td WHU_intrada_anchor --save 1 --unfreeze 25  \
#       --bs 16 --anchor --source_train_txt easy_example.txt --target_train_txt hard_example.txt \
#       --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt


#   intrada + anchor + pix
# python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth --sd Clouds26 --td WHU_intrada_anchor_pix --save 1 --unfreeze 25  \
#       --bs 16 --anchor --pixcontrast --source_train_txt easy_example.txt --target_train_txt hard_example.txt \
#       --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt


# #   intrada + anchor + pix + consist
# python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth --sd Clouds26 --td WHU_intrada_anchor_pix_consist --save 1 --unfreeze 25  \
#       --bs 16 --anchor --pixcontrast --consist --source_train_txt easy_example.txt --target_train_txt hard_example.txt \
#       --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# 关键参数消融实验

# segloss      contrastloss
#    0.5             1     key_0
#    1              0.1    key_1
#    1.5            0.01   key_2
#    2              0.001  baseline  实验已做
#    2.5            0.0001  key_3
#    3              0.00001 key_4


python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth \
      --seg_para 0.5 --contrast_para 1 \
      --sd Clouds26 --td WHU_Key_0 --save 1 --unfreeze 25  \
      --bs 16 --anchor --pixcontrast --consist \
      --source_train_txt  easy_example.txt --target_train_txt hard_example.txt \
      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt


python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth \
      --seg_para 1 --contrast_para 0.1 \
      --sd Clouds26 --td WHU_Key_1 --save 1 --unfreeze 25  \
      --bs 16 --anchor --pixcontrast --consist \
      --source_train_txt  easy_example.txt --target_train_txt hard_example.txt \
      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt


python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth \
      --seg_para 1.5 --contrast_para 0.01 \
      --sd Clouds26 --td WHU_Key_2 --save 1 --unfreeze 25  \
      --bs 16 --anchor --pixcontrast --consist \
      --source_train_txt  easy_example.txt --target_train_txt hard_example.txt \
      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt


python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth \
      --seg_para 2.5 --contrast_para 0.0001 \
      --sd Clouds26 --td WHU_Key_3 --save 1 --unfreeze 25  \
      --bs 16 --anchor --pixcontrast --consist \
      --source_train_txt  easy_example.txt --target_train_txt hard_example.txt \
      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt
    
python train_ssl.py --model_path D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step1_Clouds26_WHU\\last.pth \
      --seg_para 3 --contrast_para 0.00001 \
      --sd Clouds26 --td WHU_Key_4 --save 1 --unfreeze 25  \
      --bs 16 --anchor --pixcontrast --consist \
      --source_train_txt  easy_example.txt --target_train_txt hard_example.txt \
      --source_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\source_val.txt  --target_val_txt D:\\JinKuang\\GASKING_Expiment_DA\\target_val.txt
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#