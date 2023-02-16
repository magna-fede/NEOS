#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare source space.

@author: fm02
"""

import sys
import os
from os import path
from importlib import reload

from copy import deepcopy

import numpy as np
import pandas as pd

import mne
from mne.preprocessing import ICA, create_eog_epochs
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

import mne

print(mne.__version__)


filename = "/imaging/local/software/mne_python/set_MNE_2.7.3_FS_6.0.0_environ.py"
# for Python 3 instead of execfile
exec(compile(open(filename, "rb").read(), filename, 'exec'))

subjects_dir = config.subjects_dir


def make_source_space(sbj_id):
    """ Make Source Space. """

    ### set up source space
    src = mne.setup_source_space(str(sbj_id), spacing=config.src_spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=True)

    # check if bem directory exists, and create it if not
    bem_path = path.join(subjects_dir, str(sbj_id), 'bem')
    if not path.isdir(bem_path):
        print('Creating directory %s.' % bem_path)
        os.mkdir(bem_path)

    src_fname = path.join(subjects_dir, str(sbj_id), 'bem', str(sbj_id) + '_' + str(config.src_spacing) + '-src.fif')

    print("###\nWriting source spaces to " + src_fname + "\n###")
    mne.write_source_spaces(src_fname, src, overwrite=True)


# # get all input arguments except first
# if len(sys.argv) == 1:

#     sbj_ids = np.arange(0, len(config.map_subjects)) + 1

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]

sbj_ids = [
       #     1,
       #     2,
            3,
        #   4,
            5,
            6,
       #     7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18, 
            19, 
        #   20,
            21,
            22, 
            23, 
            24,
            26,
            27
            ]
for ss in sbj_ids:

    make_source_space(ss)

print('Done.')