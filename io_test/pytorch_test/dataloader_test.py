#!/usr/bin/env python

from __future__ import print_function, division
from torch.utils.data import DataLoader, Dataset

import h5py
import matplotlib.pyplot as plt
import random
import torch

# My own dataloader implementation
class PixelMapDataset(Dataset):
    """Pixel Map dataset."""

    def __init__(self, hdf5_file_list = 'file_list.txt'):
        """
        Args:
            hdf5_file_list (string): A file with a list of the full pathname to individual
                hdf5 files.
        """

        with open(hdf5_file_list, 'r') as f:
            self.hdf5_file_list = f.readlines()
        
        self.hdf5_len_list = []
        for fn in self.hdf5_file_list:
            hdf5_file = h5py.File(fn, "r")
            self.hdf5_len_list.append(hdf5_file['data'].shape[0])
            hdf5_file.close()
        
        self.totnevt = sum(self.hdf5_len_list)
    
    def __len__(self):
        return self.totnevt

    def __getitem__(self, idx):
        fidx = 0
        if idx >= self.totnevt:
            fidx = len(self.hdf5_len_list) - 1
            idx = self.hdf5_len_list[fidx] - 1
        for i in range(len(self.hdf5_len_list)):
            if self.hdf5_len_list[i] <= idx:
                idx = idx - self.hdf5_len_list[i]
            else:
                fidx = i
                break

        hdf5_file = h5py.File(self.hdf5_file_list[fidx], "r")
        pixelmap = hdf5_file['data'][idx][:2,]

        # determine label
        iscc = hdf5_file['interaction_mode'][idx] <= 3
        isfhc = hdf5_file['pdg'][idx] > 0

        label = 0
        if iscc and isfhc:
            label = 1
        if not iscc:
            label = 2

        sample = {'pixelmap': pixelmap, 'label': label}
        hdf5_file.close()
        return sample

ds = PixelMapDataset()

# check if GPU acceleration is available
#print(torch.cuda.get_device_name(0))

idx = random.randint(0, ds.totnevt - 1)
print(ds.__getitem__(idx)['pixelmap'].shape, ds.__getitem__(idx)['label'])
plt.imshow(ds.__getitem__(idx)['pixelmap'][0])
plt.show()