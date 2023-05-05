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

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

# %%
# Set MNE's log level to DEBUG
def plot_ICA(sbj_id):
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    for over in [
                '_ovrw',
                '',
                 '_ovrwonset'
                ]:
        for condition in ['eog', 'var', 'both']:
        
            raw_test = []   
            
            for i in range(1,6):
                raw_test.append(mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_raw.fif")))
                
            raw_test= mne.concatenate_raws(raw_test)
            raw_test.load_data()
            raw_test.info['bads'] = bad_eeg
            
            raw_test.interpolate_bads(reset_bads=True)
            
            target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
            
            event_dict = {'FRP': 999}
            epochs = mne.Epochs(raw_test, target_evts, picks=['meg', 'eeg', 'eog'],
                                tmin=-0.3, tmax=0.7, 
                                event_id=event_dict,    
                                reject=reject_criteria, flat=flat_criteria,
                                preload=True)
            
            evoked = epochs['FRP'].average()
            FRP_fig = evoked.plot_joint(title= None, 
                                        times=[0, .110, .167, .210, .266, .330, .430])
            
            for i, fig in zip(['EEG','MAG','GRAD'], FRP_fig):
                fname_fig = path.join(sbj_path, 'Figures', f'FRP_all_{i}_{condition}{over}_01Hz.png')
                fig.savefig(fname_fig)

            # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
            raw_test.filter(l_freq=0.5, h_freq=None)
            evts_blocks = target_evts.copy()
            apply_ica.plot_evoked_sensors(data=raw_test, devents=evts_blocks, comp_sel=f'{condition}{over}_05Hz')

            # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
            raw_test.filter(l_freq=1, h_freq=None)
            evts_blocks = target_evts.copy()
            apply_ica.plot_evoked_sensors(data=raw_test, devents=evts_blocks, comp_sel=f'{condition}{over}_10Hz')
        
    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 24) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    plot_ICA(ss)        