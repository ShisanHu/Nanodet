# Nanodet

数据集采用COCO2017

在config/nanodet-default.yaml下修改
指定COCO2017数据集的地址
coco_root: "/home/ma-user/work/COCO2017"
指定生成mindsRecord数据集的地址
mindrecord_dir: "/home/ma-user/work/mindRecord"

先运行create_data，将coco2017转成mindsrecord格式
python create_data.py

训练命令
python train.py

推理命令
python evaluatoy.py
