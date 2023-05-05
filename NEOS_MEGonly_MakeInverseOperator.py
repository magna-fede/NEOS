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

def make_InverseOperator(sbj_id):
    subject = str(sbj_id)
    ovr = config.ovr_procedure
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
    
    info = raw_test.info


    fwd_fname = path.join(sbj_path, subject + '_MEG-fwd.fif')
    print('Reading EEG/MEG forward solution: %s.' % fwd_fname)

    fwd_meg = mne.read_forward_solution(fwd_fname)

    fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              'MEGonly_covariancematrix_auto-cov.fif')

    print('Reading covariance matrix: %s.' % fname_cov)
    noise_cov = mne.read_cov(fname=fname_cov)



    invop_meg = mne.minimum_norm.make_inverse_operator(info, fwd_meg, noise_cov,
                                                          fixed='auto', loose=loose, depth=depth,
                                                          rank='info')

    inv_fname = path.join(sbj_path, subject + '_MEG-inv.fif')
    print('Writing EEG/MEG inverse operator: %s.' % inv_fname)
    mne.minimum_norm.write_inverse_operator(fname=inv_fname, inv=invop_meg)


if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
               21,22,23,24,25,26,27,28,29,30]


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    make_InverseOperator(ss)    
    