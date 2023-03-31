#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:59:23 2023

@author: fm02
"""
import sys
import os
from os import path

import numpy as np
import pandas as pd

import mne

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from mne.minimum_norm import (apply_inverse, apply_inverse_epochs,
                              read_inverse_operator)
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs

sbj_id = 1

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

ovr = config.ovr_procedure
ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")
method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
labels_path = path.join(config.data_path, "my_ROIs")

predictability_factors = ['Predictable', 'Unpredictable']

def compute_coherence(sbj_id):
    subject = str(sbj_id)
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
        raw_test.append(mne.io.read_raw(path.join(sbj_path,
                            f"block{i}_sss_f_ica{over}_{condition}_raw.fif"))
                        )
        
    raw_test= mne.concatenate_raws(raw_test)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
    
    raw_test.interpolate_bads(reset_bads=True)
    raw_test.filter(l_freq=0.5, h_freq=None)
            
    target_evts = mne.read_events(path.join(sbj_path,
                            config.map_subjects[sbj_id][0][-3:] + \
                            '_target_events.fif')
                                  )
            
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

    epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, 
                        tmax=tmax, reject=reject_criteria, preload=True)
    
    print(epochs)
    epochs.equalize_event_counts()
    
    inv_fname = path.join(sbj_path, subject + '_EEGMEG-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    
    stcs = dict()
    for condition in predictability_factors:
        stcs[condition] = apply_inverse_epochs(epochs[condition], inverse_operator, lambda2, method,
                                pick_ori="normal", return_generator=True)

    fmin = (8., 13.)
    fmax = (13., 30.)
    sfreq = raw_test.info['sfreq']  # the sampling frequency

    src = inverse_operator['src']

    lATL = mne.read_label(path.join(labels_path, 'l_ATL_fsaverage-lh.label'),
                          subject='fsaverage')
    rATL = mne.read_label(path.join(labels_path, 'r_ATL_fsaverage-rh.label'),
                          subject='fsaverage')
    PVA = mne.read_label(path.join(labels_path, 'PVA_fsaverage-lh.label'),
                          subject='fsaverage')
    IFG = mne.read_label(path.join(labels_path, 'IFG_fsaverage-lh.label'),
                          subject='fsaverage')
    AG = mne.read_label(path.join(labels_path, 'AG_fsaverage-lh.label'),
                          subject='fsaverage')
    PTC = mne.read_label(path.join(labels_path, 'PTC_fsaverage-lh.label'),
                          subject='fsaverage')

    times=np.arange(-300,701,1)

    rois = [lATL,
            rATL, 
            PVA,
            IFG,
            AG,
            PTC]
        
    morphed_labels = mne.morph_labels(rois, subject_to=str(sbj_id),
              subject_from='fsaverage', subjects_dir=config.subjects_dir
              )
    for condition in predictability_factors: 
        label_ts = mne.extract_label_time_course(
            stcs[condition], morphed_labels,
            src, mode='mean_flip', return_generator=True
            )
    
        coh = spectral_connectivity_epochs(
            label_ts, method='coh', mode='fourier',
            sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, n_jobs=-1
            )
    
        coh.save(path.join(sbj_path, f"{sbj_id}_{condition}_ROI_coherence"))

# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    compute_coherence(ss)
        
    