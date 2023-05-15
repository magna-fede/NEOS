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

def ovr_sub(ovr):
    if ovr in ['nover', 'novr', 'novrw']:
        ovr = ''
    elif ovr in ['ovrw', 'ovr', 'over', 'overw']:
        ovr = '_ovrw'
    elif ovr in ['ovrwonset', 'ovrons', 'overonset']:
        ovr = '_ovrwonset'
    return ovr


# %%

def make_InverseOperator(sbj_id, cov='empirical',
                         fwd='EEGMEG', inv_suf=''):
    subject = str(sbj_id)
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw_test = apply_ica.get_ica_raw(sbj_id, 
                                     condition='both',
                                     overweighting=ovr,
                                     interpolate=False, 
                                     drop_EEG_4_8=False)
    
    raw_test = raw_test.set_eeg_reference(ref_channels='average', projection=True)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
    
    if "_dropbads" in cov:
        raw_test.pick_types(meg=True, eeg=True, exclude='bads')
    else:
        raw_test.drop_channels(['EEG004', 'EEG008'])
        raw_test.interpolate_bads(reset_bads=True)
    
    info = raw_test.info


    fwd_fname = path.join(sbj_path, subject + f'_{fwd}-fwd.fif')
    print('Reading EEG/MEG forward solution: %s.' % fwd_fname)

    fwd_eegmeg = mne.read_forward_solution(fwd_fname)

    fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_covariancematrix_{cov}-cov.fif')

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
    