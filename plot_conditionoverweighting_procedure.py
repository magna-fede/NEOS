#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:01:01 2023

@author: fm02
"""

import sys
import os
from os import path
from pathlib import Path

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

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat


def plot_evoked_for_comparisons(sbj_id):
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    raws = dict()
    
    for condition in ['eog', 'var', 'both']:
        raws[condition] = dict()        
        for key in [
                    'ovrw',
                    'novr',
                     'ovrwonset'
                    ]:
            raws[condition][key] = list()
        
            for key, over in zip(raws[condition].keys(), ['_ovrw', '', '_ovrwonset']):
                raws[condition][key] = apply_ica.get_ica_raw(sbj_id, condition=condition,
                                                             overweighting=over)
    
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      '_target_events.fif'))
    
    for condition in ['eog', 'var', 'both']:
        for key, over in zip(raws[condition].keys(), ['_ovrw', '', '_ovrwonset']):
            
            event_dict = {'FRP': 999}
            epochs = mne.Epochs(raws[condition][key], target_evts, picks=['meg', 'eeg', 'eog'],
                                tmin=-0.3, tmax=0.7, 
                                event_id=event_dict,    
                                reject=reject_criteria, flat=flat_criteria,
                                preload=True)
            
            evoked = epochs['FRP'].average()
            FRP_fig = evoked.plot_joint(title= None, 
                                        times=[0, .110, .167, .210, .266, .330, .430])
            
            for i, fig in zip(['EEG','MAG','GRAD'], FRP_fig):
                fname_fig = path.join(sbj_path, 'Figures', f'FRP_all_{i}_{condition}{over}.png')
                fig.savefig(fname_fig)
                
        