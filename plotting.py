import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


FORMAT = 'png'

def read_log(path):
    data = pd.read_csv(path, sep=' ', header=None)
    data = data.iloc[:,[2,4,8,11,14]]
    data.columns = ['epoch', 'loss', 'time', 'test_acc', 'train_acc']
    return data


def savefig(filename):
    plt.savefig(
        filename + "." + FORMAT, format=FORMAT, dpi=1200, bbox_inches="tight")


def get_avg(folder, log_path, log_num):
    '''
    Calculating the averaged runing results
    '''
    avg_epoch = None
    avg_test_acc = None
    avg_train_acc = None

    for i in range(1,1+log_num):
        data = read_log(f'{folder}/{log_path}_{i}/{log_path}_{i}.txt')
        if avg_epoch is None:
            avg_epoch = data.epoch
        if avg_test_acc is None:
            avg_test_acc = data.test_acc
            avg_train_acc = data.train_acc
        else:
            avg_test_acc += data.test_acc
            avg_train_acc += data.train_acc

    avg_test_acc /= log_num
    avg_train_acc /= log_num
    avg = pd.DataFrame({'epoch':avg_epoch, 'test_acc': avg_test_acc, 'train_acc': avg_train_acc})
    
    return avg


def get_epoch(data, threshold):
    '''
    Calculating when the model's accuracy reach the threshold
    '''
    vals = data[data['test_acc']>=threshold].epoch.values
    if len(vals)!=0:
        return vals[0]
    return 'Never'


def plot_train_test(df, label):
    '''
    Plot train and test in one figure
    '''
    plt.figure(figsize=(8,6))

    plt.plot(df['epoch'], df['test_acc'], label=f'Test: {label}', color='orange')
    plt.plot(df['epoch'], df['train_acc'], label=f'Train: {label}', color='gray')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title(f'Train and Test Accuracy for {label}', fontsize=18)
    plt.legend()


def plot_tradeoff(paths, filename, acc_name='Test'):
    '''
    Plot the teachers trade off
    '''
    datas = []
    for path in paths:
        datas.append(read_log(path))
    
    name = "Pastel2"
    cmap = get_cmap(name) 
    colors = cmap.colors 


    plt.clf()
    plt.figure(figsize=(8,6))

    with plt.rc_context({"axes.prop_cycle" : plt.cycler("color", colors)}):
        for i in range(len(datas)):
            plt.plot(datas[i]['epoch'], datas[i][acc_name.lower()+'_acc'], label=f'run {i}', linewidth=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Total Trainning Time', fontsize=16)
    plt.ylabel(f'{acc_name} Accuracy', fontsize=16)
    plt.legend()

    savefig(filename)
    
if __name__ == '__main__':
    paths = []
    for i in range(1,6):
        paths.append(f'0330/5teachers_0330_logs/t{i}_0330.txt')

    datas = []
    for path in paths:
        datas.append(read_log(path))

    avg_always = get_avg('0330','kd_always_0330_5t',3)
    avg_first = get_avg('0330', 'kd_first_0330',3)
    avg_every = get_avg('0330', 'kd_every_0330',3)


    name = "Pastel2"
    cmap = get_cmap(name) 
    colors = cmap.colors 

    plt.figure(figsize=(8,6))

    with plt.rc_context({"axes.prop_cycle" : plt.cycler("color", colors)}):
        for i in range(len(datas)):
            plt.plot(datas[i]['epoch'], datas[i]['train_acc'], label=f'run {i}', linewidth=2)

    plt.plot(avg_first['epoch'], avg_first['train_acc'], label='kd_mixup_first20(avg.)', color='r')
    plt.plot(avg_always['epoch'], avg_always['train_acc'], label='kd_mixup_always(avg.)', color='b')
    plt.plot(avg_every['epoch'], avg_every['train_acc'], label='kd_mixup_every10(avg.)', color='g')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Train Accuracy', fontsize=16)
    plt.legend()

    
    