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

    def __init__(self, hdf5_file_list = 'file_list.txt', channel = 'both'):
        """
        Args:
            hdf5_file_list (string): A file with a list of the full pathname to individual
                hdf5 files.
            channel (string): Value -- 'time', 'charge', 'both', default to both
        """

        with open(hdf5_file_list, 'r') as f:
            self.hdf5_file_list = f.read().splitlines()
        
        self.hdf5_files = []
        for fn in self.hdf5_file_list:
            self.hdf5_files.append(h5py.File(fn, 'r'))

        self.hdf5_len_list = []
        for f in self.hdf5_files:
            self.hdf5_len_list.append(f['data'].shape[0])
        
        self.totnevt = sum(self.hdf5_len_list)

        # sanity check variables
        self.fidx = 0
        self.idx_in_file = 0

        # record channel(s) being used
        self.channel = channel
    
    def __len__(self):
        return self.totnevt

    def __getitem__(self, idx):
        fidx = 0 # file index
        if idx >= self.totnevt:
            fidx = len(self.hdf5_len_list) - 1
            idx = self.hdf5_len_list[fidx] - 1
        for i in range(len(self.hdf5_len_list)):
            if self.hdf5_len_list[i] <= idx:
                idx = idx - self.hdf5_len_list[i]
            else:
                fidx = i
                break
        
        # assign values to sanity check variables
        self.fidx = fidx
        self.idx_in_file = idx

        hdf5_file = self.hdf5_files[fidx]
        # even channels: charge
        # odd channels: time
        pixelmap = hdf5_file['data'][idx][:2,]
        if self.channel == 'time':
            pixelmap = hdf5_file['data'][idx][:1,]
        if self.channel == 'charge':
            pixelmap = hdf5_file['data'][idx][[0],]

        # determine label
        iscc = hdf5_file['interaction_mode'][idx] <= 3
        isnumu = (hdf5_file['pdg'][idx] == 14)
        isnumubar = (hdf5_file['pdg'][idx] == -14)

        # signel (numubar cc): 0
        # background1 (numu cc): 1
        # background2 (all others): 2
        label = 2
        if iscc and isnumubar:
            label = 0
        if iscc and isnumu:
            label = 1

        # sample = {'pixelmap': pixelmap, 'label': label}
        # return sample
        return pixelmap, label