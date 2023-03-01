#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:58:21 2022

@author: fm02
adapted from py01 EyeCA
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
from my_eyeCA import preprocess, ica, snr_metrics

os.chdir("/home/fm02/MEG_NEOS/NEOS")

mne.viz.set_browser_backend("matplotlib")

# %%
# Set MNE's log level to DEBUG
mne.set_log_level(verbose="DEBUG")

sbj_id = int(sys.argv[1])

sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

# raw-filename mappings for this subject
tmp_fnames = config.sss_map_fnames[sbj_id][1]

# only use files for correct conditions
sss_map_fnames = []
for sss_file in tmp_fnames:
    sss_map_fnames.append(sss_file)


for over in [
             '_ovrw',
             '',
             '_ovrwonset'
            ]:
    data_raw_files = []
    
    # load unfiltered data to fit ICA with
    for raw_stem_in in sss_map_fnames:
        data_raw_files.append(
            path.join(path.join(sbj_path, raw_stem_in[:6] +
                                f"_sss_f_ica{over}_both_raw.fif")))
    
    bad_eeg = config.bad_channels[sbj_id]['eeg']
    
    
    	# %%
    for block, drf in enumerate(data_raw_files):
        raw = mne.io.read_raw(drf, preload=True)
        raw.info['bads'] = bad_eeg
        
        raw_orig = raw.copy()
        
        evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  f'_all_events_block_{block+1}.fif')
        evt = mne.read_events(evt_file)
        
        ica01 = snr_metrics.compute_metrics(raw, evt, plot=False)
        ica01["type"] = "0.1Hz"
        
        # %% High-pass raw data at 0.5Hz & compute SNR metrics on ICA-reconstructed data
        raw.filter(l_freq=0.5, h_freq=None)
        ica05 = snr_metrics.compute_metrics(raw, evt, plot=False)
        ica05["type"] = "0.5Hz"
        
        # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
        raw.filter(l_freq=1.0, h_freq=None)
        ica10 = snr_metrics.compute_metrics(raw, evt, plot=False)
        ica10["type"] = "1.0Hz"
        # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
        raw.filter(l_freq=2.0, h_freq=None)
        ica20 = snr_metrics.compute_metrics(raw, evt, plot=False)
        ica20["type"] = "2.0Hz"
        # %% Visualize differences in SNR metrics
        df = pd.DataFrame([ica01, ica05, ica10, ica20])
        cols = {
            "P1_SNR": "P1 SNR",
            "GFP_first100": "GFP (Baseline)",
            "GFP_n400": "GFP (N400)",
            "GFP_last100": "GFP (Last 100ms)",
            "SNR_n400": "GFP Ratio (n400/Bsaline)",
            "S_amplitude": "Saccade Peak Ampl.",
            "S_auc": "Saccade AUC",
        }
        order = ["0.1Hz", "0.5Hz", "1.0Hz", "2.0Hz"]
        palette = dict(zip(order, ["dodgerblue", "lightsalmon", "yellowgreen", "plum"]))
        
        fig, axx = plt.subplots(1, len(cols), figsize=(20, 5), tight_layout=True)
        for m, col in enumerate(cols.keys()):
        
            ax = axx[m]
            sns.barplot(data=df, x="type", y=col, ax=ax, order=order, palette=palette)
            ax.set_title(cols[col])
            ax.set_xlabel("")
            ax.set_ylabel("")
        
        sns.despine(fig)
        fname_fig = path.join(sbj_path, 'Figures', f'snr_metrics_ICA_filtering_{sbj_id}_{block+1}{over}_both.png')
        fig.savefig(fname_fig) 
        
        df.to_csv(path.join(sbj_path, f"snr_compare_filt_{sbj_id}_{block+1}{over}_both.csv"))
