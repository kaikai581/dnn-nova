#!/usr/bin/env python

from __future__ import print_function, division
from io_helpers import *
from models.alexnet import alexnet

import torch

# hyperparameters
batch_size = 64

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# restore model
model = alexnet(num_classes=3, in_ch=2).to(device)
model.load_state_dict(torch.load('../../../data/saved_models/alexnet/model.ckpt'))

# load test dataset
test_dataset = PixelMapDataset('test_file_list.txt')

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
    for images, labels in test_loader:
        images = images.float().to(device)
        labels = labels.to(device)
        outputs = model(images)
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
    print('3-class Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    print('2-class Test Accuracy of the model on the test images: {} %'.format(100 * correct_cc_or_bkg / total))
    print('Wrong-sign Test Accuracy of the model on the test images: {} %'.format(100 * ws_correct / ws_total))