#!/usr/bin/env python

from __future__ import print_function, division
from io_helpers import *
from models.alexnet import alexnet
from models.densenet import DenseNet
from models.inception import inception_v3
from models.resnet import resnet18
from models.squeezenet import squeezenet1_1

import os
import torch
import torch.nn as nn

def train_model(modname = 'alexnet', pm_ch = 'both', bs = 16):
    """
    Args:
        modname (string): Name of the model. Has to be one of the values:
            'alexnet', batch 64
            'densenet'
            'inception'
            'resnet', batch 16
            'squeezenet', batch 16
        pm_ch (string): pixelmap channel -- 'time', 'charge', 'both', default to both
    """
    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # hyper parameters
    max_epochs = 10
    learning_rate = 0.001

    # determine number of input channels
    nch = 2
    if pm_ch != 'both':
        nch = 1

    ds = PixelMapDataset('training_file_list.txt', pm_ch)
    # try out the data loader utility
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=bs, shuffle=True)

    # define model
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
    else:
        print('Model {} not defined.'.format(modname))
        return

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training process
    total_step = len(dl)
    for epoch in range(max_epochs):
        for i, (view1, view2, local_labels) in enumerate(dl):
            view1 = view1.float().to(device)
            view1 = nn.ZeroPad2d((0,117,64,64))(view1)
            local_labels = local_labels.to(device)

            # forward pass
            outputs = model(view1)
            loss = criterion(outputs, local_labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % bs == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

    # save the model checkpoint
    save_path = '../../../data/two_views/saved_models/{}/{}'.format(modname, pm_ch)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.ckpt'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='alexnet', type=str)
    parser.add_argument('-c', '--pixelmap_channel', default='both', type=str)
    parser.add_argument('-b', '--batch_size', default=16, type=int)

    args = parser.parse_args()

    train_model(args.model_name, args.pixelmap_channel, args.batch_size)