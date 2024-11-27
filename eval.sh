#------------------------------------------------------------------------#
# ours
# 95 -> WHU
# python vis_image.py --dir D:\\JinKuang\\images\\WHU_95 \
#        --save_path 95_WHU --method ours  \
#        --m D:\\JinKuang\\UDS\\Pth\\25.pth  --is_write

# Clouds26 -> WHU
# python vis_image.py --dir D:\\JinKuang\\images\\Clouds26_WHU \
#        --save_path Clouds26_WHU --method ours  \
#        --m D:\\JinKuang\\UDS\\Pth\\entropy\\step3_Clouds26_WHU\\last.pth  --is_write

# WHU -> Clouds26
# python vis_image.py --dir D:\\JinKuang\\images\\WHU_Clouds26 \
#        --save_path WHU_Clouds26 --method ours  \
#        --m D:\\JinKuang\\UDS\\Pth\\entropy\\step3_WHU_Clouds26\\5.pth  --is_write




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#    关键参数消融实验


python vis_image.py --dir D:\\JinKuang\\images\\ablation_study \
       --save_path Clouds26_WHU_intrada_anchor_pix --method ours  \
       --m D:\\JinKuang\\UDS\\Pth\\step3_Clouds26_WHU_intrada_anchor_pix\\last.pth  --is_write


python vis_image.py --dir D:\\JinKuang\\images\\ablation_study \
       --save_path Clouds26_WHU_intrada_anchor_pix_consist --method ours  \
       --m D:\\JinKuang\\UDS\\Pth\\paper\\entropy\\step3_Clouds26_WHU\\last.pth  --is_write

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#