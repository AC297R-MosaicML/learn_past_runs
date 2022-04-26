#!/bin/bash

model=resnet56
dataset=CIFAR100
# data_dir='drive/MyDrive/Colab_Notebooks/'
data_dir='../'
save_dir='drive/MyDrive/Colab_Notebooks/test_save_dir'
epochs=5
logname1='noteacher_01'
logname2='noteacher_05'

echo "python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir/$logname1 --epochs=$epochs --log-name=$logname1"
python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir/$logname1 --epochs=$epochs --log-name=$logname1

echo "python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir/$logname2 --epochs=$epochs --log-name=$logname2 --lr=0.5"
python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir/$logname2 --epochs=$epochs --log-name=$logname2 --lr=0.5
