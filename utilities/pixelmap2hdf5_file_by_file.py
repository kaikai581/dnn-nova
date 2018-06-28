#!/usr/bin/env python
# $ setup_nova -b maxopt -r R17-11-14-prod4reco.neutron-respin.b
###############################################################################
# NEVER FORGET TO SETUP THE TEST RELEASE!!!!
###############################################################################
# $ cd /nova/ana/users/slin/test_releases/wrong_sign_cnn_R17-11-14-prod4reco.neutron-respin.b
# $ srt_setup -a

from __future__ import print_function

import argparse
import glob
import os
import subprocess
import sys

# command line arguments dictating the begin and end files
parser = argparse.ArgumentParser()
parser.add_argument('--begin', '-b', type=int, default=0)
parser.add_argument('--end', '-e', type=int, default=sys.maxint)
args = parser.parse_args()

# in_dir = '/pnfs/nova/users/slin/wrong_sign_cnn/event_dump_root_files'
out_dir = '/pnfs/nova/users/slin/wrong_sign_cnn/pixel_map_hdf5_files'
binary = '/nova/ana/users/slin/test_releases/wrong_sign_cnn_R17-11-14-prod4reco.neutron-respin.b/bin/Linux2.6-GCC-maxopt/pixelMap2Hdf5'

# inf_list = [name for name in glob.glob('{}/*.root'.format(in_dir))]
file_list = '/nova/ana/users/slin/temp/cvn_test/event_dump_round2/test_input.list'
inf_list = []
with open(file_list) as fl:
    for f in fl: inf_list.append(f)

# trim input size if larger than there are
args.end = min(args.end, len(inf_list)-1)

# loop through all files
for i in range(args.begin, args.end+1):
    out_fn_test = os.path.join(out_dir, 'test_'+os.path.basename(inf_list[i]).rstrip('\n').rstrip('.root')+'.h5')
    out_fn_training = os.path.join(out_dir, 'training_'+os.path.basename(inf_list[i]).rstrip('\n').rstrip('.root')+'.h5')
    print('Processing the {}th file:\n\t{}'.format(i, inf_list[i]))
    cmd = 'time {} -i {}'.format(binary, inf_list[i])
    subprocess.call(cmd, shell=True)
    subprocess.call('mv -v test_data.h5 '+out_fn_test, shell=True)
    subprocess.call('mv -v training_data.h5 '+out_fn_training, shell=True)