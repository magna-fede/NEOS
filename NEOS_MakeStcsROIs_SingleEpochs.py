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


import mne
from mne.minimum_norm import apply_inverse_epochs

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

#os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import preprocess, ica, snr_metrics, apply_ica

os.chdir("/home/fm02/MEG_NEOS/NEOS")

mne.viz.set_browser_backend("matplotlib")


snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
loose = 0.2
depth = None
reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

labels_path = path.join(config.data_path, "my_ROIs")
stc_path = path.join(config.data_path, "stcs")

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

# %%

def make_stcsEpochs(sbj_id, method='eLORETA', inv_suf='emp3150'):
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
    
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG-inv_{inv_suf}.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)

    rois_lab = ['lATL',
                'rATL', 
                'PVA',
                'IFG',
                'AG',
                'PTC']
    

    stc= dict()
    for i, roi in enumerate(rois):
        stc[rois_lab[i]] = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, roi,
                                pick_ori="normal", nave=len(epochs))
    for roi in rois_lab:
        stc_fname = path.join(stc_path, f"{subject}_stc_{roi}_{method}")
        stc[roi].save(stc_fname)
        
        !!!! does not work, you lose info about epoch ID this way
        it would be better to estimate timecourses for each condition
        for each roi separately, and save that date:
            that way you can also read it in R and do a mixed model.

# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]


# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     make_InverseOperator(ss)    
    