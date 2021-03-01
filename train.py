#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import random
import time
from tqdm import tqdm

import torch, torchvision
from torch.utils import data
#import torch.nn.functional as F
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from models.resnet import resnet_face18, resnet34, resnet50, resnet_face34
from models.metrics import ArcMarginProduct, AddMarginProduct, SphereProduct
from models.focal_loss import FocalLoss
from config.config import Config
from data.dataset import Dataset

from eval import evaluate
#from test import get_lfw_list, lfw_test

from utils import view_model
from utils.visualizer import Visualizer


# global value to record the best accuracy
best_acc = 0.0


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def train(opt, model, metric_fc, device, train_loader, loss_func, optimizer, lr_scheduler):
    train_loss = 0.0
    correct = 0.0
    model.train()

    # decay learning rate every epoch
    if lr_scheduler:
        lr_scheduler.step()

    tbar = tqdm(train_loader)
    for i, (data, target) in enumerate(tbar):
        # forward propagation
        data, target = data.to(device), target.to(device).long()
        feature = model(data)
        output = metric_fc(feature, target) if not isinstance(metric_fc, torch.nn.Linear) else metric_fc(feature)

        # calculate loss
        #output = F.log_softmax(output, dim=1)
        #loss = nn.NLLLoss()(output, target)
        #loss = F.nll_loss(output, target)
        #loss = nn.CrossEntropyLoss()(output, target)
        #loss = FocalLoss(gamma=2)(output, target)
        #loss = F.cross_entropy(output, target)
        loss = loss_func(output, target)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # collect loss and accuracy
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        tbar.set_description('Train loss: %06.4f - acc: %06.4f' % (train_loss/(i + 1), correct/((i + 1)*opt.train_batch_size)))



def validate(opt, model, device, epoch, log_dir):
    global best_acc
    val_acc, val_threshold = evaluate(model, 'PTH', opt.input_shape[1:], device, opt.lfw_root, opt.lfw_test_list)

    # save checkpoint with best accuracy
    if val_acc > best_acc:
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, 'ep{epoch:03d}-val_acc{val_acc:.3f}-val_threshold{val_threshold:.3f}.pth'.format(epoch=epoch+1, val_acc=val_acc, val_threshold=val_threshold))
        torch.save(model, checkpoint_dir)
        print('Epoch {epoch:03d}: val_acc improved from {best_acc:.3f} to {val_acc:.3f}, saving model to {checkpoint_dir}'.format(epoch=epoch+1, best_acc=best_acc, val_acc=val_acc, checkpoint_dir=checkpoint_dir))
        best_acc = val_acc
    else:
        print('Epoch {epoch:03d}: val_acc did not improve from {best_acc:.3f}'.format(epoch=epoch+1, best_acc=best_acc))


def main():
    log_dir = os.path.join('logs', '000')
    opt = Config()
    if opt.display:
        visualizer = Visualizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1)

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    #identity_list = get_lfw_list(opt.lfw_test_list)
    #img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]


    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        #model = resnet34()
        model = resnet_face34(use_se=opt.use_se)
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35, device=device)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin, device=device)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4, device=device)
    else:
        metric_fc = torch.nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    #print(model)
    model.to(device)
    summary(model, input_size=opt.input_shape)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    #start = time.time()
    for epoch in range(opt.max_epoch):
        scheduler.step()
        print('Epoch %d/%d'%(epoch, opt.max_epoch))
        train(opt, model, metric_fc, device, trainloader, criterion, optimizer, scheduler)
        validate(opt, model, device, epoch, log_dir)

"""

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            os.makedirs(opt.checkpoints_path, exist_ok=True)
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        #acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        acc, _ = evaluate(model, 'PTH', opt.input_shape[1:], device, opt.lfw_root, opt.lfw_test_list)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')
"""


if __name__ == '__main__':
    main()
