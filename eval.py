import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from models import *
from metrics import *
from utils import *


def validate(val_loader, model, criterion, set_name, device):

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()  

            correct += compute_correct(output, target)

    test_loss /= len(val_loader.dataset)
    acc = correct.float() / len(val_loader.dataset)

    print('Evaluate on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        set_name, test_loss, correct, len(val_loader.dataset),
        100. * acc))

    return test_loss, acc


def caching(val_loader, teacher_models, set_name, device, model_masks):
    print(f'Cache teachers output on {set_name}')
    for idx, model in enumerate(teacher_models):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
        
                _, preds = output.max(1)
                masking = preds.eq(target)
                correct += masking.sum()
                model_masks['teacher_'+str(idx)].append(masking)

        acc = correct.float() / len(val_loader.dataset)
        assert len(model_masks['teacher_'+str(idx)]) == len(val_loader), "masking size not equal to dataloader!"
        print('Evaluate on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            set_name, test_loss, correct, len(val_loader.dataset),
            100. * acc))

    return model_masks