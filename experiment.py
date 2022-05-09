'''
We refer to [1] for parsing arguments and main()

Reference:
[1] https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
'''


import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from train import train, train_masking
from eval import validate, caching
from dataset import load_dataset
from utils import Tradeoff


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                    help='model architecture: resnet56 |resnet110')
parser.add_argument('--data', default='CIFAR100',
                    help='Dataset: CIFAR10 | CIFAR100')
parser.add_argument('--data_dir', default='./data',
                    help='The directory of the dataset')
parser.add_argument('--download', default=False,
                    help='whether to download the dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--teacher', default='', type=str, metavar='PATH',
                    help='path to the teacher model (default: none)')
parser.add_argument('--teacher_num', default=5, type=int, help='teacher_num')
parser.add_argument('--tr', '--teacher_random', type=int, default=5, help='random sample # teachers per batch')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-freq', dest='save_freq',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)
parser.add_argument('--log-name', default='default_log',  help='log file name')
parser.add_argument('--use-cuda', type=bool,  default=True, help='if use cuda') # Maybe change back to default=True later
parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--kd_epochs_first', type=int, default=200, help='use kd for the first several epochs')
parser.add_argument('--kd_epochs_every', type=int, default=1, help='use kd every x epochs')
parser.add_argument('--random_weights', action='store_true', help='if assign random weights to teacher models')
parser.add_argument('--mask', type=bool,  default=False, help='if research on good/bad')


best_prec1 = 0


def main(args, best_prec1):

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda" if args.use_cuda else "cpu")

    if args.data.lower() == 'cifar10':
        num_classes = 10
    elif args.data.lower() == 'cifar100':
        num_classes = 100

    model = torch.nn.DataParallel(models.__dict__[args.arch](num_classes=num_classes))
    model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    
    # load one teacher model
    if args.teacher:
        # is a path, one teacher
        if os.path.isfile(args.teacher):
            teacher_model = torch.nn.DataParallel(models.__dict__[args.arch](num_classes=num_classes))
            checkpoint = torch.load(args.teacher)
            teacher_model.load_state_dict(checkpoint['state_dict'])
            teacher_model = [teacher_model]
            st_criterion = nn.MSELoss().to(device)
        # is a folder, loading several teachers
        else:
            teacher_model = []
            st_criterion = nn.MSELoss().to(device)
            paths = sorted(os.listdir(args.teacher))
            for i, path in enumerate(paths):
                if i < args.teacher_num:
                    one_model = torch.nn.DataParallel(models.__dict__[args.arch](num_classes=num_classes))
                    checkpoint = torch.load(os.path.join(args.teacher, path))
                    one_model.load_state_dict(checkpoint['state_dict'])
                    teacher_model.append(one_model)
            print('Total {} teachers, they are {}, later we will subsample {} teachers per batch'.format(len(teacher_model), paths[:args.teacher_num], args.tr))
        assert args.tr <= len(teacher_model), 'random sample # teachers per batch is larger than teacher total number!'
        if args.mask:
            t_masks = {}
            for i in range(len(teacher_model)):
                t_masks["teacher_"+str(i)] = []
    else:
        teacher_model = None
        st_criterion = None


    cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        load_dataset(args.data.lower(), args.data_dir, 'train', args.download),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        load_dataset(args.data.lower(), args.data_dir, 'test', args.download),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    eval_train = torch.utils.data.DataLoader(
        load_dataset(args.data.lower(), args.data_dir, 'eval_train', args.download),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                         milestones=[100, 150], 
#                                                         last_epoch=args.start_epoch - 1)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
#     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    tradeoff = Tradeoff()
    if teacher_model and args.mask:
        model_masks = caching(eval_train, teacher_model, 'train data', device, model_masks)

    for epoch in range(args.start_epoch, args.epochs + 1):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        if epoch <= args.kd_epochs_first and epoch % args.kd_epochs_every == 0:
            if args.mask:
                train_loss, train_time = train_masking(train_loader, model, criterion, optimizer, 
                                           epoch, device, args.print_freq, st_criterion, 
                                           teacher_model, args.lambda_kd, model_masks)
            else:
                train_loss, train_time = train(train_loader, model, criterion, optimizer, 
                                           epoch, device, args.print_freq, st_criterion, 
                                           teacher_model, args.lambda_kd, args.tr, 
                                           args.random_weights)
        else:
            train_loss, train_time = train(train_loader, model, criterion, optimizer, epoch, device, args.print_freq)

        lr_scheduler.step()

        # evaluate
        test_loss, test_acc = validate(val_loader, model, criterion, 'test data', device)
        eval_train_loss, eval_train_acc = validate(eval_train, model, criterion, 'train data', device)
        tradeoff.update(train_time, test_acc)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(test_acc, best_prec1)

        log_tmp = 'Train Epoch: {} Loss: {:.6f} Total Training time: {:.2f} Test Accuracy: {:.3f} Train Accuracy: {:.3f}\n'.format(
            epoch, train_loss, tradeoff.train_time,  test_acc, eval_train_acc)
        with open(os.path.join(args.save_dir,"{}.txt".format(args.log_name)), "a") as log:
            log.write(log_tmp)
        print(log_tmp)

        if epoch > 0 and epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, os.path.join(args.save_dir, 'epoch_{}.th'.format(epoch)))


if __name__ == '__main__':
    main(parser.parse_args(), best_prec1)