#!/usr/bin/env python
# reference:
# http://machinelearninguru.com/deep_learning/data_preparation/hdf5/hdf5.html

from __future__ import print_function

import h5py
import matplotlib.pyplot as plt
import numpy as numpy
import random

hdf5_pathname = '../pixelmap_hdf5/test_neardet_genie_nonswap_genierw_rhc_v08_950_r00012116_s16_c023_R17-11-14-prod4reco.neutron-respin.b_v1_20170802_144926_sim.evt_dump.h5'

hdf5_file = h5py.File(hdf5_pathname, "r")

# print shape for each stored data
for key in hdf5_file.keys(): print(key, hdf5_file[key].shape, type(hdf5_file[key][0]))

# show the first image
plt.imshow(hdf5_file['data'][random.randint(0,hdf5_file['data'].shape[0]-1)][random.randint(0, 3)])
plt.show()

hdf5_file.close()