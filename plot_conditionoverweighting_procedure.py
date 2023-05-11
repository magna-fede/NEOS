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
    

    for condition in ['eog', 'variance', 'both']:
        snr_ica = dict()
        for key, over in zip(["nover", "ovrw", "ovrwonset"], ['', '_ovrw', '_ovrwonset']):
            raws = apply_ica.get_ica_raw(sbj_id, condition=condition,
                                         overweighting=over, drop_EEG_4_8=False)
            evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      '_all_events.fif'))
            snr_ica[key] = snr_metrics.compute_metrics(raws.copy(), evts, plot=False)
            
            target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      '_target_events.fif'))
    
            event_dict = {'FRP': 999}
            epochs = mne.Epochs(raws, target_evts, picks=['meg', 'eeg', 'eog'],
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
            
        df = pd.DataFrame(snr_ica).T
        cols = {
            "P1_SNR": "P1 SNR",
            "GFP_first100": "GFP (first 100 ms)",
            "GFP_baseline": "GFP (Baseline)",
            "GFP_fixation_onset": "GFP (Fixation Onset)",
            "GFP_late": "GFP (last 300 ms)",
            "S_amplitude": "Saccade Peak Ampl.",
            "S_auc": "Saccade AUC",
        }
        order = ["nover", "ovrw", "ovrwonset"]
        palette = dict(zip(order, ["dodgerblue","yellowgreen", "plum"]))
        
        fig, axx = plt.subplots(1, len(cols), figsize=(20, 5), tight_layout=True)
        
        for m, col in enumerate(cols.keys()):
        
            ax = axx[m]
            sns.barplot(data=df, x=["nover", "ovrw", "ovrwonset"], y=col, ax=ax, order=order, palette=palette)
            ax.set_title(cols[col])
            ax.set_xlabel("")
            ax.set_ylabel("")
        
        sns.despine(fig)
        fname_fig = path.join(sbj_path, 'Figures', f'snr_metrics_ICA_componentselection_{sbj_id}_{condition}.png')
        fig.savefig(fname_fig) 
           
           
        df.to_csv(path.join(sbj_path, f"snr_compareoverweight_{sbj_id}_{condition}.csv"))

        