#!/usr/bin/env python

from __future__ import print_function, division
from io_helpers import *
from models.alexnet import alexnet
from models.densenet import DenseNet
from models.inception import inception_v3
from models.resnet import resnet18
from models.squeezenet import squeezenet1_1
from models.vgg import vgg19_bn

import os
import torch
import torch.nn as nn

def test_model(modname = 'alexnet', pm_ch = 'both', bs = 16):
    # hyperparameters
    batch_size = bs

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # determine number of input channels
    nch = 2
    if pm_ch != 'both':
        nch = 1

    # restore model
    model = None
    if modname == 'alexnet':
        model = alexnet(num_classes=3, in_ch=nch).to(device)
    elif modname == 'densenet':
        model = DenseNet(num_classes=3, in_ch=nch).to(device)
    elif modname == 'inception':
        model = inception_v3(num_classes=3, in_ch=nch).to(device)
    elif modname == 'resnet':
        model = resnet18(num_classes=3, in_ch=nch).to(device)
    elif modname == 'squeezenet':
        model = squeezenet1_1(num_classes=3, in_ch=nch).to(device)
    elif modname == 'vgg':
        model = vgg19_bn(in_ch=nch, num_classes=3).to(device)
    else:
        print('Model {} not defined.'.format(modname))
        return
    
    # retrieve trained model
    # load path
    load_path = '../../../data/two_views/saved_models/{}/{}'.format(modname, pm_ch)
    model_pathname = os.path.join(load_path, 'model.ckpt')
    if not os.path.exists(model_pathname):
        print('Trained model file {} does not exist. Abort.'.format(model_pathname))
        return
    model.load_state_dict(torch.load(model_pathname))

    # load test dataset
    test_dataset = PixelMapDataset('test_file_list.txt', pm_ch)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        correct_cc_or_bkg = 0
        ws_total = 0
        ws_correct = 0
        for view1, view2, labels in test_loader:
            view1 = view1.float().to(device)
            if modname == 'inception':
                view1 = nn.ZeroPad2d((0,192,102,101))(view1)
            else:
                view1 = nn.ZeroPad2d((0,117,64,64))(view1)
            labels = labels.to(device)
            outputs = model(view1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                if (predicted[i] < 2 and labels[i] < 2) or (predicted[i] == 2 and labels[i] == 2):
                    correct_cc_or_bkg += 1
                if labels[i] < 2:
                    ws_total += 1
                    if (predicted[i] == labels[i]):
                        ws_correct += 1
        print('Model Performance:')
        print('Model:', modname)
        print('Channel:', pm_ch)
        print('3-class Test Accuracy of the model on the test images: {}/{}, {:.2f} %'.format(correct, total, 100 * correct / total))
        print('2-class Test Accuracy of the model on the test images: {}/{}, {:.2f} %'.format(correct_cc_or_bkg, total, 100 * correct_cc_or_bkg / total))
        print('Wrong-sign Test Accuracy of the model on the test images: {}/{}, {:.2f} %'.format(ws_correct, ws_total, 100 * ws_correct / ws_total))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='alexnet', type=str)
    parser.add_argument('-c', '--pixelmap_channel', default='both', type=str)
    parser.add_argument('-b', '--batch_size', default=16, type=int)

    args = parser.parse_args()

    test_model(args.model_name, args.pixelmap_channel, args.batch_size)