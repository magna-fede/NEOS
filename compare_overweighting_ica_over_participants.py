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

os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import preprocess, ica, metrics, apply_ica

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

data_raw_files = []

# load unfiltered data to fit ICA with
for raw_stem_in in sss_map_fnames:
    data_raw_files.append(
        path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_raw.fif'))

bad_eeg = config.bad_channels[sbj_id]['eeg']


	# %%
for block, drf in enumerate(data_raw_files):
    raw = mne.io.read_raw(drf)
    raw.info['bads'] = bad_eeg
    
    raw_orig = raw.copy()
    
    evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_all_events_block_{block+1}.fif')
    evt = mne.read_events(evt_file)
    
    evt_xy_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_all_events_xy_block_{block+1}.csv')
    evt_xy = pd.read_csv(evt_xy_file)

    # call function that does the overweighting
    raw_ica = preprocess.overweight_saccades(raw, evt_xy)
    
	# %%
    ic = ica.run_ica_pipeline(
	    raw=raw_ica, evt=evt, method="extinfomax", cov_estimator=None, n_comp=0.99, drf=drf
	)
    pre_ica, figs = metrics.compute_metrics(raw_orig, evt, plot=True)
    pre_ica["type"] = "pre-ICA"
    
    # %%
    variance_threshold = 1.1

	# %%
    raw_recon = raw_orig.copy()
    raw_eogica, ic_eog, ic_eog_scores = apply_ica.apply_ica_pipeline(raw=raw_recon,                                                                  
                            evt=evt, thresh=variance_threshold,
    						ica_instance=ic, method='eog', ovrw=True)
    
    # %% Compute SNR metrics on ICA-reconstructed data
    eog_ica, _ = metrics.compute_metrics(raw_eogica, evt, plot=True)
    eog_ica["type"] = "eog"
    
    # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
    
    raw_recon = raw_orig.copy()
    raw_varica, ic_var, ic_var_scores = apply_ica.apply_ica_pipeline(raw=raw_recon,                                                                  
                            evt=evt, thresh=variance_threshold,
    						ica_instance=ic, method='variance', ovrw=True)
    
    # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
    var_ica, _ = metrics.compute_metrics(raw_varica, evt, plot=True)
    var_ica["type"] = "variance"
    
    # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
    
    raw_recon = raw_orig.copy()
    raw_bothica, ic_both, ic_both_scores = apply_ica.apply_ica_pipeline(raw=raw_recon,                                                                  
                            evt=evt, thresh=variance_threshold,
    						ica_instance=ic, method='both', ovrw=True)
    
    # %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
    both_ica, _ = metrics.compute_metrics(raw_bothica, evt, plot=True)
    both_ica["type"] = "both"
              
    # %% Visualize differences in SNR metrics
    df = pd.DataFrame([pre_ica, eog_ica, var_ica, both_ica])
    cols = {
        "P1_amplitude": "P1 Peak Ampl.",
        "GFP_first100": "GFP (Baseline)",
        "GFP_n400": "GFP (N400)",
        "GFP_last100": "GFP (Last 100ms)",
        "GFP_ratio": "GFP Ratio (L/B)",
        "S_amplitude": "Saccade Peak Ampl.",
        "S_auc": "Saccade AUC",
    }
    order = ["pre-ICA", "eog", "variance", "both"]
    palette = dict(zip(order, ["dodgerblue", "lightsalmon", "yellowgreen", "plum"]))
    
    fig, axx = plt.subplots(1, len(cols), figsize=(20, 5), tight_layout=True)
    
    for m, col in enumerate(cols.keys()):
    
        ax = axx[m]
        sns.barplot(data=df, x="type", y=col, ax=ax, order=order, palette=palette)
        ax.set_title(cols[col])
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    sns.despine(fig)
    fname_fig = path.join(sbj_path, 'Figures', f'metrics_ICA_filtering_{sbj_id}_{block}_overweight.png')
    fig.savefig(fname_fig) 