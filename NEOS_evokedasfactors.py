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

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

ovr = config.ovr_procedure

# %%
# Set MNE's log level to DEBUG
def create_evoked(sbj_id):
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels[sbj_id]['eeg']
    
    if ovr[sbj_id] == 'ovrons':
        over = '_ovrwonset'
    elif ovr[sbj_id] == 'ovr':
        over = '_ovrw'
    elif ovr[sbj_id] == 'novr':
        over = ''
    condition = 'both'

    raw_test = []   
            
    for i in range(1,6):
        raw_test.append(mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_ica{over}_{condition}_raw.fif")))
        
    raw_test= mne.concatenate_raws(raw_test)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
    
    raw_test.interpolate_bads(reset_bads=True)
    raw_test.filter(l_freq=0.5, h_freq=None)
            
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
    
    # cond1 = 'Predictable'
    # cond2 = 'Unpredictable'
    

    # evoked_pred = epochs[cond1].average()
    # mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_predictable.fif", evoked_pred, overwrite=True)
    
    # evoked_unpred = epochs[cond2].average()
    # mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_unpredictable.fif", evoked_unpred, overwrite=True)
    
    # cond1 = 'Concrete'
    # cond2 = 'Abstract'
    
    # evoked_conc = epochs[cond1].average()
    # mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_concrete.fif", evoked_conc, overwrite=True)
    
    # evoked_abs = epochs[cond2].average()
    # mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_abstract.fif", evoked_abs, overwrite=True)
    
    cond1 = 'Concrete/Predictable'
    cond2 = 'Abstract/Predictable'
    
    evoked_conc = epochs[cond1].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_concpred.fif", evoked_conc, overwrite=True)
    
    evoked_abs = epochs[cond2].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_abspred.fif", evoked_abs, overwrite=True)
    
    cond1 = 'Concrete/Unpredictable'
    cond2 = 'Abstract/Unpredictable'
    
    evoked_conc = epochs[cond1].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_concunpred.fif", evoked_conc, overwrite=True)
    
    evoked_abs = epochs[cond2].average()
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked_absunpred.fif", evoked_abs, overwrite=True)    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 30) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    create_evoked(ss)        
    
    