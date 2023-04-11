#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:15 2023

@author: fm02
"""

import matplotlib.pyplot as plt

import mne
from mne.stats.regression import linear_regression_raw


import sys
import os
from os import path

import numpy as np
import pandas as pd

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from my_eyeCA import apply_ica

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

ovr = config.ovr_procedure

# %%
# Set MNE's log level to DEBUG
def create_evoked(sbj_id):
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    if ovr[sbj_id] == 'ovrons':
        over = '_ovrwonset'
        ica_dir = path.join(sbj_path, 'ICA_ovr_w_onset')
    elif ovr[sbj_id] == 'ovr':
        over = '_ovrw'
        ica_dir = path.join(sbj_path, 'ICA_ovr_w')
    elif ovr[sbj_id] == 'novr':
        over = ''
        ica_dir = path.join(sbj_path, 'ICA')
    condition = 'both'

    raw_test = []
    
    ica_sub = '_sss_f_raw_ICA_extinfomax_0.99_COV_None'
    ica_sub_file = '_sss_f_raw_ICA_extinfomax_0.99_COV_None-ica_eogvar'
            
    for i in range(1, 6):
        ica_fname = path.join(ica_dir,
                              f'block{i}'+ica_sub,
                              f'block{i}'+ica_sub_file) 
        raw = mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_raw.fif"))
        evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          f'_all_events_block_{i}.fif')
        evt = mne.read_events(evt_file)
        # hey, very important to keep overwrite_saved as false, or this will change
        # the raw files saved for checking which approach is best for each participant
        raw_ica, _, _ = apply_ica.apply_ica_pipeline(raw=raw,                                                                  
                        evt=evt, thresh=1.1, method='both',
						ica_filename=ica_fname, overwrite_saved=False)
        raw_test.append(raw_ica)
    raw_test = mne.concatenate_raws(raw_test)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
    
    raw_test.interpolate_bads(reset_bads=True)
    raw_test.drop_channels(['EEG004', 'EEG008'])
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
            
    rows = np.where(target_evts[:,2]==999)[0]
    for row in rows:
        if target_evts[row-2, 2] == 1:
            target_evts[row, 2] = 991
        elif target_evts[row-2, 2] == 2:
            target_evts[row, 2] = 992
        elif target_evts[row-2, 2] == 3:
            target_evts[row, 2] = 993
        elif target_evts[row-2, 2] == 4:
            target_evts[row, 2] = 994
        elif target_evts[row-2, 2] == 5:
            target_evts[row, 2] = 995
            
    event_dict = {'Abstract/Predictable': 991, 
                  'Concrete/Predictable': 992,
                  'Abstract/Unpredictable': 993, 
                  'Concrete/Unpredictable': 994}
    tmin, tmax = -.3, .7
    
    # regular epoching

    epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
                        reject=reject_criteria, preload=True)

    # evokeds = linear_regression_raw(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
    #                     reject=reject_criteria)

    evokeds = list()    

    for key in event_dict.keys():
        evokeds.append(epochs[key].average())
    
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evokeds_nochan_4_8.fif", evokeds, overwrite=True)

# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 30) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    create_evoked(ss)        
    
    