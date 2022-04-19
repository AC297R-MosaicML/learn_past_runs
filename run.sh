#!/bin/bash

model=resnet56
dataset=CIFAR100
# data_dir='drive/MyDrive/Colab_Notebooks/'
data_dir='../datasets'
save_dir='../logs/test_save_dir'
teacher="../../5teachers_0330"
logname="CGH20220414_0"

echo "python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir \
  --save-dir=$save_dir --ST_criterion=KL --log-name=$logname --teacher=$teacher"

python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir --ST_criterion=KL --log-name=$logname --teacher=$teacher
