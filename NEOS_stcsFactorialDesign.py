#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:51:16 2023

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

import re


ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")
snr = 3.
lambda2 = 1. / snr ** 2
# orientation = 'normal'
conditions = [['Predictable', 'Unpredictable'], ['Abstract', 'Concrete']]
# conditions = ['abspred', 'absunpred', 'concpred', 'concunpred']


def compute_stcs(sbj_id, method="MNE", inv_suf='', orientation=None):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    evs=dict()
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evokeds_nochan_4_8.fif")
        evokeds = mne.read_evokeds(fname)
        ev_0 = list()
        ev_1 = list()
        for e, evoked in enumerate(evokeds):
            if re.match(f'([a-z]*\/)?{condition[0]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_0.append(evoked)
            elif re.match(f'([a-z]*\/)?{condition[1]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_1.append(evoked)  
                
        evs[condition[0]] = mne.combine_evoked(ev_0, weights='nave')
        evs[condition[1]] = mne.combine_evoked(ev_1, weights='nave')
        
    for ev in evs.keys():
        stc = mne.minimum_norm.apply_inverse(evs[ev], inverse_operator,
                                             lambda2, method=method,
                                             pick_ori=orientation, verbose=True)
        if len(inv_suf)==0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}")
        elif len(inv_suf)>0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}_{inv_suf}")
        stc.save(stc_fname)
        
def compute_stcs_dropbads(sbj_id, method="MNE", inv_suf='', orientation=None):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    evs=dict()
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evokeds_dropbads.fif")
        evokeds = mne.read_evokeds(fname)
        ev_0 = list()
        ev_1 = list()
        for e, evoked in enumerate(evokeds):
            if re.match(f'([a-z]*\/)?{condition[0]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_0.append(evoked)
            elif re.match(f'([a-z]*\/)?{condition[1]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_1.append(evoked)  
                
        evs[condition[0]] = mne.combine_evoked(ev_0, weights='nave')
        evs[condition[1]] = mne.combine_evoked(ev_1, weights='nave')
        
    for ev in evs.keys():
        stc = mne.minimum_norm.apply_inverse(evs[ev], inverse_operator,
                                             lambda2, method=method,
                                             pick_ori=orientation, verbose=True)
        if len(inv_suf)==0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}")
        elif len(inv_suf)>0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}_{inv_suf}")
        stc.save(stc_fname)


# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]


# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     compute_stcs(ss)    
    