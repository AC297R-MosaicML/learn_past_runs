import time

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


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq, st_criterion=None, t_model=None):

    model.train()

    start = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()

        # TO DO: further check and debugging
        s_out = model(data)
        if t_model:
            for param in t_model.parameters():
                param.requires_grad = False
            t_out = t_model(data)

        cls_loss = criterion(s_out, target)
        st_loss = st_criterion(s_out, t_out) if st_criterion else 0

        loss = cls_loss + st_loss
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    end = time.time()

    return loss, end-start