#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ICA corrected F/ERPs across all trials (althuogh correction is per block)
@author: fm02
"""

import sys
import os
from os import path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import mne
from mne.preprocessing import ICA, create_eog_epochs
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

# os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import preprocess, ica, snr_metrics, apply_ica

os.chdir("/home/fm02/MEG_NEOS/NEOS")



# %%
# Set MNE's log level to DEBUG
def plot_ICA(sbj_id):
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    for over in ['_ovrw', '']:
        for condition in ['eog', 'var', 'both']:
        
            raw_test = []   
            
            for i in range(1,6):
                raw_test.append(mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_ica{over}_{condition}_raw.fif")))
                
            raw_test= mne.concatenate_raws(raw_test)
            raw_test.load_data()
            
            target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_target_events.fif'))
            
            event_dict = {'FRP': 999}
            epochs = mne.Epochs(raw_test, target_evts, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,                   
                        preload=True)
            evoked = epochs['FRP'].average()
            FRP_fig = evoked.plot_joint(times=[0, .110, .167, .210, .266, .330, .430])
            
            for i, fig in zip(['EEG','MAG','GRAD'], FRP_fig):
                fname_fig = path.join(sbj_path, 'Figures', f'FRP_{i}_all_{condition}{over}.png')
                fig.savefig(fname_fig)     
        
    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 24) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    plot_ICA(ss)        