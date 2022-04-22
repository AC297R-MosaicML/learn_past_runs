import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from models import *
from metrics import *
from utils import *


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, 
          st_criterion=None, t_models=None, lambda_kd=1, kd_prop=1, kd_int=1):

    model.train()

    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        s_out = model(data)
        cls_loss = criterion(s_out, target)
        # loss = cls_loss

        if batch_idx % kd_int == 0 and np.random.uniform() < kd_prop and t_models:
            t_outputs = None
            frac = 1/len(t_models)
            for t_model in t_models:
                for param in t_model.parameters():
                    param.requires_grad = False
                if t_outputs is None:
                    t_outputs = t_model(data) * frac
                else:
                    t_outputs += t_model(data) * frac
#                 loss += st_criterion(s_out, t_model(data)) * lambda_kd
            loss = (kd_loss_weight*st_criterion(s_out, t_outputs) + 
                    (1-kd_loss_weight)*cls_loss)

        else:
            loss = cls_loss
    
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    end = time.time()

    return loss, end-start