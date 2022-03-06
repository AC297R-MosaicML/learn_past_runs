#!/bin/bash

model=resnet56
dataset=CIFAR100
# data_dir='drive/MyDrive/Colab_Notebooks/'
data_dir='../'
save_dir='drive/MyDrive/Colab_Notebooks/test_save_dir'

echo "python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir"
python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=$save_dir
