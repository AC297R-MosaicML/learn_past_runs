#!/bin/bash

model=resnet56
dataset=CIFAR100
data_dir='./data'

echo "python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=save_$model"
python -u experiment.py  --arch=$model  --data=$dataset --data_dir=$data_dir  --save-dir=save_$model
