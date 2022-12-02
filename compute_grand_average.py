#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:31:05 2022

@author: federica.magnabosco@mrc-cbu.cam.ac.uk
"""

import NEOS_config as config
import sys
import os
from os import path
import numpy as np
import pandas as pd

from importlib import reload
import pickle
import mne

import seaborn as sns
import matplotlib
matplotlib.use('Agg')  #  for running graphics on cluster ### EDIT

import matplotlib.pyplot as plt

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

reload(config)


reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

mne.viz.set_browser_backend("matplotlib")

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")

def evoked_sensors(sbj_id):
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]
    
    # only use files for correct conditions
    sss_map_fnames = []
    for sss_file in tmp_fnames:
        sss_map_fnames.append(sss_file)
    
    data_raw_file = path.join(sbj_path,
                               sbj_path[-3:] +
                              "_sss_f_ica_od_unfiltered_onraw_raw.fif")
    
    print(f'Reading raw file {sss_map_fnames}')
    data = mne.io.read_raw_fif(data_raw_file)
    
    devents = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_target_events.fif'))
    event_dict = {'FRP': 999}
    epochs = mne.Epochs(data, devents, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                    reject=reject_criteria, flat=flat_criteria,
                    preload=True)
        
    rows = np.where(devents[:,2]==999)
    for row in rows[0]:
        if devents[row-2, 2] == 1:
            devents[row, 2] = 991
        elif devents[row-2, 2] == 2:
            devents[row, 2] = 992
        elif devents[row-2, 2] == 3:
            devents[row, 2] = 993
        elif devents[row-2, 2] == 4:
            devents[row, 2] = 994
        elif devents[row-2, 2] == 5:
            devents[row, 2] = 995
            
    event_dict = {'Abstract/Predictable': 991, 
                  'Concrete/Predictable': 992,
                  'Abstract/Unpredictable': 993, 
                  'Concrete/Unpredictable': 994}
    
    epochs = mne.Epochs(data, devents, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                        reject=reject_criteria, flat=flat_criteria,
                        preload=True)
  
    cond1 = 'Predictable'
    cond2 = 'Unpredictable'
    

    evoked_pred = epochs[cond1].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}evoked_predictable.fif", evoked_pred, overwrite=True)
    
    evoked_unpred = epochs[cond2].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}evoked_unpredictable.fif", evoked_unpred, overwrite=True)
    
    cond1 = 'Concrete'
    cond2 = 'Abstract'
    
    evoked_conc = epochs[cond1].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}evoked_concrete.fif", evoked_conc, overwrite=True)
    
    evoked_abs = epochs[cond2].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}evoked_abstract.fif", evoked_abs, overwrite=True)
    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = [22,23,24]

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    evoked_sensors(ss)
        













