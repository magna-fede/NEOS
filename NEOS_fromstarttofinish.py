#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collection gets you from Maxfiltered raw data to source timecourses.

NB: This ignores the steps for calculatin the most optimal ICA procedure for each subject.
That will need to be run and inferred separately.

@author: federica.magnabosco@mrc-cbu.cam.ac.uk
"""

import sys
import os
from os import path
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

import mne

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)


#%% This script takes care of fixing electrodes position, possibly redundant, but no harm in running it

from NEOS_fix_electrodes import run_fix_electrodes

# %% This script takes care of filtering and saving the filtered data

from NEOS_filter_raw import run_filter_raw
# %% These scripts synch the MEG and ET data, either per block or concatenating each block

from NEOS_synch_per_block import synchronise as synchronise_per_block
from NEOS_synchronisation_includeEDF import synchronise_concat
   
# %%
from NEOS_applyICA_evoked import create_evoked
# %% THIS will run through all the scripts above, careful is going to take time

if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
               21,22,23,24,25,26,27,28,29,30]

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    run_fix_electrodes(ss)
    run_filter_raw(ss)    
    synchronise_per_block(ss)
    synchronise_concat(ss)
    create_evoked(ss)
    
# %%
