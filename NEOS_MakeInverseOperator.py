#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:40:24 2023

@author: fm02
"""
import sys
import os
from os import path

import numpy as np
import pandas as pd

import mne
from mne.preprocessing import ICA, create_eog_epochs
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

#os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import preprocess, ica, snr_metrics, apply_ica

os.chdir("/home/fm02/MEG_NEOS/NEOS")

mne.viz.set_browser_backend("matplotlib")

# make inverse operator
loose = 0.2
depth = None

# %%

def make_InverseOperator(sbj_id, cov='covariancematrix_empirical_350150',
                         fwd='EEGMEG', inv_suf=''):
    subject = str(sbj_id)
    
    ovr = config.ovr_procedure
    
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
    
    # pick_types operates in place
    raw_test.pick_types(meg=True, eeg=True, exclude='bads')
    info = raw_test.info


    fwd_fname = path.join(sbj_path, subject + f'_{fwd}-fwd.fif')
    print('Reading EEG/MEG forward solution: %s.' % fwd_fname)

    fwd_eegmeg = mne.read_forward_solution(fwd_fname)

    fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_{cov}-cov.fif')

    print('Reading covariance matrix: %s.' % fname_cov)
    noise_cov = mne.read_cov(fname=fname_cov)



    invop_eegmeg = mne.minimum_norm.make_inverse_operator(info, fwd_eegmeg, noise_cov,
                                                          fixed='auto', loose=loose, depth=depth,
                                                          rank='info')

    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    print('Writing EEG/MEG inverse operator: %s.' % inv_fname)
    mne.minimum_norm.write_inverse_operator(fname=inv_fname, inv=invop_eegmeg)


# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]


# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     make_InverseOperator(ss)    
    