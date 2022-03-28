from logging import raiseExceptions
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100


def load_dataset(data, data_dir, split, download):
    assert split in ["train", "test"], f"unknown split: {split}"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if data[:5].lower() == 'cifar':
        cifar_sets = {
            "cifar10": CIFAR10,
            "cifar100": CIFAR100,
        }

        if split == 'train':
            return cifar_sets[data](root=data_dir, train=True, transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]), download=download)
        else:
            return cifar_sets[data](root=data_dir, train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), download=download)
    else:
        raiseExceptions('Only support cifar datasets!')