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


ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")
method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2
orientation=None
# orientation = 'normal'
conditions = ['predictable', 'unpredictable', 'abstract', 'concrete']
# conditions = ['abspred', 'absunpred', 'concpred', 'concunpred']


def compute_stcs(sbj_id):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + '_EEGMEG-inv_solved.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evoked_{condition}.fif")
        evoked = mne.read_evokeds(fname)
            
        stc = mne.minimum_norm.apply_inverse(evoked[0], inverse_operator,
                                             lambda2, method=method,
                                             pick_ori=orientation, verbose=True)
        stc_fname = path.join(stc_path, f"{subject}_stc_{condition}_solved")
        stc.save(stc_fname)

if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
               21,22,23,24,25,26,27,28,29,30]


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    compute_stcs(ss)    
    