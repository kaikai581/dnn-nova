#!/usr/bin/env python

from __future__ import print_function, division
from io_helpers import *
from models.alexnet import alexnet

import torch
import torch.nn as nn

# device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyper parameters
max_epochs = 10
learning_rate = 0.001

ds = PixelMapDataset('training_file_list.txt')
# try out the data loader utility
dl = torch.utils.data.DataLoader(dataset=ds, batch_size=64, shuffle=True)

# define model
model = alexnet(num_classes=3, in_ch=2).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training process
total_step = len(dl)
for epoch in range(max_epochs):
    for i, (local_batch, local_labels) in enumerate(dl):
        local_batch = local_batch.float().to(device)
        local_labels = local_labels.to(device)

        # forward pass
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

# save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')