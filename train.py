import time
from random import sample
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import *
from metrics import *
from utils import *
import numpy as np

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, st_criterion=None, t_models=None, lambda_kd=1, t_num=5, random_weights=False):
    model.train()
    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        s_out = model(data)
        cls_loss = criterion(s_out, target)
        loss = cls_loss
        
        if t_models and t_num > 0:
            t_total = len(t_models)
            t_outputs = None
            if random_weights is False:
                # randomly sample a subset of teachers per batch
                sel_idx = set(sample(range(t_total), t_num))
                frac = 1 / t_num
                for i, t_model in enumerate(t_models):
                    if i in sel_idx:
                        for param in t_model.parameters():
                            param.requires_grad = False
                        if t_outputs is None:
                            t_outputs = t_model(data) * frac
                        else:
                            t_outputs += t_model(data) * frac
            else:
#                 weights = np.random.dirichlet(np.ones(len(t_models)),size=1)
                weights = np.random.rand(len(t_models))
                weights = weights / np.sum(weights)
                for idx, t_model in enumerate(t_models):
                    for param in t_model.parameters():
                        param.requires_grad = False
                    if t_outputs is None:
                        t_outputs = t_model(data) * weights[idx]
                    else:
                        t_outputs += t_model(data) * weights[idx]

#             loss += st_criterion(s_out, t_model(data)) * lambda_kd
            loss += st_criterion(s_out, t_outputs) * lambda_kd
    
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    end = time.time()
    
    return loss, end-start