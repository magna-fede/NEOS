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


# %%


max_list = dict(eeg=[], mag=[], grad=[])
min_list = dict(eeg=[], mag=[], grad=[])
avg_list = dict(eeg=[], mag=[], grad=[])

def check_frps(sbj_id):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + '_EEGMEG-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evoked_{condition}.fif")
        evoked = mne.read_evokeds(fname)
        picks_eeg = mne.pick_types(evoked[0].info, exclude='bads', meg=False, eeg=True)
        picks_grad = mne.pick_types(evoked[0].info, exclude='bads', meg='grad', eeg=False)
        picks_mag = mne.pick_types(evoked[0].info, exclude='bads', meg='mag', eeg=False)
            
        for ch_type, picks in zip(['eeg', 'mag', 'grad'], [picks_eeg, picks_mag, picks_grad]):
            fig = evoked[0].copy().pick(picks).plot()
            fname_fig = f'/imaging/hauk/users/fm02/MEG_NEOS/data/misc/check_frps/{subject}_{ch_type}_{condition}.png'
            fig.savefig(fname_fig)
            onetype = evoked[0].copy().pick(picks).get_data()
            max_list[ch_type].append(onetype.max())
            min_list[ch_type].append(onetype.min())
            avg_list[ch_type].append(onetype.mean())
            
    save_max = pd.DataFrame.from_dict(max_list)
    save_max.to_csv(f'/imaging/hauk/users/fm02/MEG_NEOS/data/misc/check_frps/{subject}_max.csv',
                    index=False)
    save_min = pd.DataFrame.from_dict(min_list)   
    save_min.to_csv(f'/imaging/hauk/users/fm02/MEG_NEOS/data/misc/check_frps/{subject}_min.csv',
                    index=False)
    save_avg = pd.DataFrame.from_dict(avg_list)
    save_avg.to_csv(f'/imaging/hauk/users/fm02/MEG_NEOS/data/misc/check_frps/{subject}_avg.csv',
                    index=False)
    
if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
               21,22,23,24,25,26,27,28,29,30]


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    check_frps(ss)    
    