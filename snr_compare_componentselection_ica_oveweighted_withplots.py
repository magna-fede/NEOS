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

from my_eyeCA import preprocess, ica, snr_metrics, apply_ica

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
    # data has been already filtered, interpolated, and average-referenced
    data_raw_files.append(
        path.join(sbj_path, raw_stem_in[:-7] + 'sss_raw.fif'))

data_filtered_files = []
# load unfiltered data to fit ICA with
for raw_stem_in in sss_map_fnames:
    # data has been already filtered, interpolated, and average-referenced
    data_filtered_files.append(
        path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_raw.fif'))

bad_eeg = config.bad_channels_all[sbj_id]['eeg'].copy()

# for computing ICA, channel 4 and 8 are useful, so although we mark them as bad to exclude
# them from inversion (i.e., covariance is very often too high from either of them)
# we want to keep them when computing the ICA, as this might help in identifying
# eye artefacts

[bad_eeg.remove(too_close_to_the_eyes) for too_close_to_the_eyes in ['EEG004', 'EEG008']]

    # %%
for block, drf in enumerate(data_raw_files):
    raw = mne.io.read_raw(drf, preload=True)

    raw = raw.pick_types(meg=True, eeg=True, eog=True, stim=True, 
                         ecg=False, emg=False)

    print('Fixing coil types.')
    raw.fix_mag_coil_types()

    raw.info['bads'] = bad_eeg

    print('Setting EEG reference.')
    raw.set_eeg_reference(ref_channels='average', projection=True)

    raw_orig = raw.copy()
        
    evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_all_events_block_{block+1}.fif')
    evt = mne.read_events(evt_file)
    
    evt_xy_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_all_events_xy_block_{block+1}.csv')
    evt_xy = pd.read_csv(evt_xy_file)

    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      f'_target_events_block_{block+1}.fif'))
   
    # call function that does the overweighting
    raw_ica = preprocess.overweight_saccades(raw, evt_xy)

    # ICA solution will be applied to filtered data
    raw_filt = mne.io.read_raw(data_filtered_files[block], preload = True)
    raw_filt.info['bads'] = bad_eeg
    raw_filt.set_eeg_reference(ref_channels='average', projection=True)

    raw_recon = raw_filt.copy()

	# %%
    ic = ica.run_ica_pipeline(
	    raw=raw_ica, evt=evt, method="extinfomax", cov_estimator=None,
        n_comp=0.99, over_type='ovrw', drf=drf
	)
    pre_ica, figs = snr_metrics.compute_metrics(raw_recon, evt, standard_rejection=False, plot=True)
    pre_ica["type"] = "pre-ICA"

    apply_ica.plot_evoked_sensors(data=raw_recon, devents=target_evts,
                                  comp_sel=f'_{block+1}_pre-ICA_overweight',
                                  standard_rejection=False)       
    
    # %%
    variance_threshold = 1.1

	# %%
    raw_recon = raw_filt.copy()
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      f'_target_events_block_{block+1}.fif'))
    raw_eogica, ic_eog, ic_eog_scores = apply_ica.apply_ica_pipeline(raw=raw_recon,                                                                  
                            evt=evt, thresh=variance_threshold, plot_overlay=True,
    						ica_instance=ic, method='eog', over='_ovrw')
    apply_ica.plot_evoked_sensors(data=raw_eogica, devents=target_evts, 
                                  comp_sel=f'_{block+1}_eog_overweight',
                                  standard_rejection=True)    
    # %% Compute SNR snr_metrics on ICA-reconstructed data
    eog_ica, _ = snr_metrics.compute_metrics(raw_eogica, evt, plot=True)
    eog_ica["type"] = "eog"

    
    # %% High-pass raw data at 1Hz & compute SNR snr_metrics on ICA-reconstructed data
    
    raw_recon = raw_filt.copy()
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      f'_target_events_block_{block+1}.fif'))
    raw_varica, ic_var, ic_var_scores = apply_ica.apply_ica_pipeline(raw=raw_recon,                                                                  
                            evt=evt, thresh=variance_threshold, plot_overlay=True,
    						ica_instance=ic, method='variance', over='_ovrw')
    apply_ica.plot_evoked_sensors(data=raw_varica, devents=target_evts, 
                                  comp_sel=f'_{block+1}_variance_overweight',
                                  standard_rejection=True)
    # %% High-pass raw data at 1Hz & compute SNR snr_metrics on ICA-reconstructed data
    var_ica, _ = snr_metrics.compute_metrics(raw_varica, evt, plot=True)
    var_ica["type"] = "variance"


    # %% High-pass raw data at 1Hz & compute SNR snr_metrics on ICA-reconstructed data
    
    raw_recon = raw_filt.copy()
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      f'_target_events_block_{block+1}.fif'))
    raw_bothica, ic_both, ic_both_scores = apply_ica.apply_ica_pipeline(raw=raw_recon,                                                                  
                            evt=evt, thresh=variance_threshold, plot_overlay=True,
    						ica_instance=ic, method='both',  over='_ovrw')
    apply_ica.plot_evoked_sensors(data=raw_bothica, devents=target_evts, 
                                  comp_sel=f'_{block+1}_both_overweight',
                                  standard_rejection=True)
    # %% High-pass raw data at 1Hz & compute SNR snr_metrics on ICA-reconstructed data
    both_ica, _ = snr_metrics.compute_metrics(raw_bothica, evt, plot=True)
    both_ica["type"] = "both"

              
    # %% Visualize differences in SNR snr_metrics
    df = pd.DataFrame([pre_ica, eog_ica, var_ica, both_ica])
    cols = {
        "P1_SNR": "P1 SNR",
        "GFP_first100": "GFP (first 100 ms)",
        "GFP_baseline": "GFP (Baseline)",
        "GFP_fixation_onset": "GFP (Fixation Onset)",
        "GFP_late": "GFP (last 300 ms)",
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
    fname_fig = path.join(sbj_path, 'Figures', f'snr_metrics_ICA_componentselection_{sbj_id}_{block+1}_overweight.png')
    fig.savefig(fname_fig) 


    df.to_csv(path.join(sbj_path, f"snr_componentselection_{sbj_id}_{block+1}_overweight.csv"))

