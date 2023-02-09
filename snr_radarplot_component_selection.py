#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created summary for overweight vs NOoverweight and component selection.
Average across blocks.
Plot all participants to check decision is stable across participants.

@author: fm02
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

ovr_metrics = dict()
novr_metrics = dict()

for sbj_id in config.do_subjs:
    
    sbj_metrics_ovr = pd.DataFrame()
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    for i in range(5):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_componentselection_{sbj_id}_{i+1}_overweight.csv")
                                  )
        sbj_metrics_ovr = pd.concat([sbj_metrics_ovr, blk_metrics], ignore_index=True)
    
    sbj_metrics_ovr = sbj_metrics_ovr.drop("Unnamed: 0", axis=1)
    sbj_metrics_ovr = sbj_metrics_ovr.groupby(["type"]).mean()
    
    ovr_metrics[sbj_id] = sbj_metrics_ovr
    
    sbj_metrics_novr = pd.DataFrame()
    
    for i in range(5):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_componentselection_{sbj_id}_{i+1}_NOoverweight.csv")
                                  )
        sbj_metrics_novr = pd.concat([sbj_metrics_novr, blk_metrics], ignore_index=True)
    
    sbj_metrics_novr = sbj_metrics_novr.drop("Unnamed: 0", axis=1)
    sbj_metrics_novr = sbj_metrics_novr.groupby(["type"]).mean()
    novr_metrics[sbj_id] = sbj_metrics_novr

df_novr = pd.concat(novr_metrics.values(), keys=novr_metrics.keys())
df_ovr = pd.concat(ovr_metrics.values(), keys=ovr_metrics.keys())
    
# categories = ['P1_SNR', 'P1_latency', 'GFP_first100', 'GFP_n400', 'GFP_last100',
#                'SNR_n400', 'S_amplitude', 'S_latency', 'S_auc']
# magn = [1/10, 10, 10000000, 1000000, 1000000, 1, 1000000, 100, 10000000]
categories = ['P1_SNR', 'GFP_first100', 'GFP_n400', 
               'SNR_n400','S_amplitude']

magn = [1/10, 10000000, 1000000, 1, 1000000]

categories = [*categories, categories[0]]



for sbj_id in config.do_subjs:
    OVR_both = np.multiply(df_ovr[categories].loc[sbj_id,"both"].values[:-1], magn)
    OVR_eog = np.multiply(df_ovr[categories].loc[sbj_id,"eog"].values[:-1], magn)
    #OVR_preICA = np.multiply(df_ovr[categories].loc[sbj_id,"pre-ICA"].values[:-1], magn)
    OVR_var = np.multiply(df_ovr[categories].loc[sbj_id,"variance"].values[:-1], magn)
    
    
    nOVR_both = np.multiply(df_novr[categories].loc[sbj_id,"both"].values[:-1], magn)
    nOVR_eog = np.multiply(df_novr[categories].loc[sbj_id,"eog"].values[:-1], magn)
    #nOVR_preICA = np.multiply(df_novr[categories].loc[sbj_id,"pre-ICA"].values[:-1], magn)
    nOVR_var = np.multiply(df_novr[categories].loc[sbj_id,"variance"].values[:-1], magn)
    
    
    OVR_both = [*OVR_both, OVR_both[0]]
    OVR_eog = [*OVR_eog, OVR_eog[0]]
    #OVR_preICA = [*OVR_preICA, OVR_preICA[0]]
    OVR_var = [*OVR_var, OVR_var[0]]
    
    nOVR_both = [*nOVR_both, nOVR_both[0]]
    nOVR_eog = [*nOVR_eog, nOVR_eog[0]]
    #nOVR_preICA = [*nOVR_preICA, nOVR_preICA[0]]
    nOVR_var = [*nOVR_var, nOVR_var[0]]
    
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    plt.figure(figsize=(12,12))
    plt.subplot(polar=True)
    plt.plot(label_loc, OVR_both, label='Overweighted - Both')
    plt.plot(label_loc, OVR_eog, label='Overweighted - EOG')
    #plt.plot(label_loc, OVR_preICA, label='Overweighted - preICA')
    plt.plot(label_loc, OVR_var, label='Overweighted - Variance')
    plt.plot(label_loc, nOVR_both, label='Non-overweighted - Both')
    plt.plot(label_loc, nOVR_eog, label='Non-overweighted - EOG')
    #plt.plot(label_loc, nOVR_preICA, label='Non-overweighted - preICA')
    plt.plot(label_loc, nOVR_var, label='Non-overweighted - Variance')
    plt.fill(label_loc, OVR_both, label='Overweighted - Both', alpha=0.1)
    plt.title(f'Participant {sbj_id}', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    plt.show()


categories = ['P1_SNR', 'SNR_n400','S_amplitude','S_auc']
categories = [*categories, categories[0]]
magn = [1, 1, 1000000, 10000000]
for sbj_id in config.do_subjs:
    OVR_both = np.multiply(df_ovr[categories].loc[sbj_id,"both"].values[:-1], magn)
    OVR_eog = np.multiply(df_ovr[categories].loc[sbj_id,"eog"].values[:-1], magn)
    OVR_preICA = np.multiply(df_ovr[categories].loc[sbj_id,"pre-ICA"].values[:-1], magn)
    OVR_var = np.multiply(df_ovr[categories].loc[sbj_id,"variance"].values[:-1], magn)
    
    
    nOVR_both = np.multiply(df_novr[categories].loc[sbj_id,"both"].values[:-1], magn)
    nOVR_eog = np.multiply(df_novr[categories].loc[sbj_id,"eog"].values[:-1], magn)
#    nOVR_preICA = np.multiply(df_novr[categories].loc[sbj_id,"pre-ICA"].values[:-1], magn)
    nOVR_var = np.multiply(df_novr[categories].loc[sbj_id,"variance"].values[:-1], magn)
    
    
    OVR_both = [*OVR_both, OVR_both[0]]
    OVR_eog = [*OVR_eog, OVR_eog[0]]
    OVR_preICA = [*OVR_preICA, OVR_preICA[0]]
    OVR_var = [*OVR_var, OVR_var[0]]
    
    nOVR_both = [*nOVR_both, nOVR_both[0]]
    nOVR_eog = [*nOVR_eog, nOVR_eog[0]]
#    nOVR_preICA = [*nOVR_preICA, nOVR_preICA[0]]
    nOVR_var = [*nOVR_var, nOVR_var[0]]
    
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    plt.figure(figsize=(12,12))
    plt.subplot(polar=True)
    plt.plot(label_loc, OVR_both, label='Overweighted - Both')
    plt.plot(label_loc, OVR_eog, label='Overweighted - EOG')
    plt.plot(label_loc, OVR_preICA, label='Overweighted - preICA')
    plt.plot(label_loc, OVR_var, label='Overweighted - Variance')
    plt.plot(label_loc, nOVR_both, label='Non-overweighted - Both')
    plt.plot(label_loc, nOVR_eog, label='Non-overweighted - EOG')
#    plt.plot(label_loc, nOVR_preICA, label='Non-overweighted - preICA')
    plt.plot(label_loc, nOVR_var, label='Non-overweighted - Variance')
    plt.fill(label_loc, OVR_both, label='Overweighted - Both', alpha=0.1)
    plt.title(f'Participant {sbj_id}', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    plt.show()
    

# average SNR

    OVR_both = np.multiply(df_ovr[categories].loc[(slice(None), "both"), :].mean(axis=0).values[:-1], magn) 
    OVR_eog = np.multiply(df_ovr[categories].loc[(slice(None), "eog"), :].mean(axis=0).values[:-1], magn) 
    OVR_preICA = np.multiply(df_ovr[categories].loc[(slice(None), "pre-ICA"), :].mean(axis=0).values[:-1], magn) 
    OVR_var = np.multiply(df_ovr[categories].loc[(slice(None), "variance"), :].mean(axis=0).values[:-1], magn) 
    
    
    nOVR_both = np.multiply(df_novr[categories].loc[(slice(None), "both"), :].mean(axis=0).values[:-1], magn) 
    nOVR_eog = np.multiply(df_novr[categories].loc[(slice(None), "eog"), :].mean(axis=0).values[:-1], magn) 
#    nOVR_preICA = np.multiply(df_novr[categories].loc[(slice(None), "pre-ICA"), :].mean(axis=0)
    nOVR_var = np.multiply(df_novr[categories].loc[(slice(None), "variance"), :].mean(axis=0).values[:-1], magn) 
    
    OVR_both = [*OVR_both, OVR_both[0]]
    OVR_eog = [*OVR_eog, OVR_eog[0]]
    OVR_preICA = [*OVR_preICA, OVR_preICA[0]]
    OVR_var = [*OVR_var, OVR_var[0]]
    
    nOVR_both = [*nOVR_both, nOVR_both[0]]
    nOVR_eog = [*nOVR_eog, nOVR_eog[0]]
#    nOVR_preICA = [*nOVR_preICA, nOVR_preICA[0]]
    nOVR_var = [*nOVR_var, nOVR_var[0]]
    
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    plt.figure(figsize=(12,12))
    plt.subplot(polar=True)
    plt.plot(label_loc, OVR_both, label='Overweighted - Both')
    plt.plot(label_loc, OVR_eog, label='Overweighted - EOG')
    plt.plot(label_loc, OVR_preICA, label='preICA')
    plt.plot(label_loc, OVR_var, label='Overweighted - Variance')
    plt.plot(label_loc, nOVR_both, label='Non-overweighted - Both')
    plt.plot(label_loc, nOVR_eog, label='Non-overweighted - EOG')
#    plt.plot(label_loc, nOVR_preICA, label='Non-overweighted - preICA')
    plt.plot(label_loc, nOVR_var, label='Non-overweighted - Variance')
    plt.fill(label_loc, OVR_both, label='Overweighted - Both', alpha=0.1)
    plt.title('Average across participants', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    plt.show()
    
summary_ovr = df_ovr[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["both", "variance", "eog", "pre-ICA"]), :]
summary_novr = df_novr[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["both", "variance", "eog", "pre-ICA"]), :]


summary_novr.groupby(["type"]).mean()
summary_ovr.groupby(["type"]).mean()
