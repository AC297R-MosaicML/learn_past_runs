import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


FORMAT = 'png'

def read_log(path):
    data = pd.read_csv(path, sep=' ', header=None)
    data = data.iloc[:,[2,4,8,11]]
    data.columns = ['epoch', 'loss', 'time', 'acc']
    return data


def savefig(filename):
    plt.savefig(
        filename + "." + FORMAT, format=FORMAT, dpi=1200, bbox_inches="tight")


def plot_tradeoff(paths, filename):
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
            plt.plot(datas[i]['time'], datas[i]['acc'], label=f'run {i}', linewidth=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Total Trainning Time', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()

    savefig(filename)