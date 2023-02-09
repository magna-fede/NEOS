#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:47:42 2023

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
                                            f"snr_compare_filt_{sbj_id}_{block+1}_overweight_both.csv")
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

df_ovr = pd.concat(ovr_metrics.values(), keys=ovr_metrics.keys())

categories = ['P1_SNR', 'GFP_first100', 'GFP_n400', 
               'SNR_n400','S_amplitude']

magn = [1/10, 10000000, 1000000, 1, 1000000]

categories = [*categories, categories[0]]



for sbj_id in config.do_subjs:
    OVR_01 = np.multiply(df_ovr[categories].loc[sbj_id,"0.1Hz"].values[:-1], magn)
    OVR_05 = np.multiply(df_ovr[categories].loc[sbj_id,"0.5Hz"].values[:-1], magn)
    OVR_10 = np.multiply(df_ovr[categories].loc[sbj_id,"1.0Hz"].values[:-1], magn)
    OVR_20 = np.multiply(df_ovr[categories].loc[sbj_id,"2.0Hz"].values[:-1], magn)
    

    OVR_01 = [*OVR_01, OVR_01[0]]
    OVR_05 = [*OVR_05, OVR_05[0]]
    OVR_10 = [*OVR_10, OVR_10[0]]
    OVR_20 = [*OVR_20, OVR_20[0]]

    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    plt.figure(figsize=(12,12))
    plt.subplot(polar=True)
    plt.plot(label_loc, OVR_01, label='Overweighted - 0.1 Hz')
    plt.plot(label_loc, OVR_05, label='Overweighted - 0.5 Hz')
    plt.plot(label_loc, OVR_10, label='Overweighted - 1 Hz')
    plt.plot(label_loc, OVR_20, label='Overweighted - 2 Hz')
    # plt.fill(label_loc, OVR_01, label='Overweighted - Both', alpha=0.1)
    plt.title(f'Participant {sbj_id}', size=20, y=1.05)
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend()
    plt.show()


categories = ['P1_SNR', 'SNR_n400','S_amplitude','S_auc']
categories = [*categories, categories[0]]
magn = [1, 1, 1000000, 100000000]

# average SNR

OVR_01 = np.multiply(df_ovr[categories].loc[(slice(None), "0.1Hz"), :].mean(axis=0).values[:-1], magn) 
OVR_05 = np.multiply(df_ovr[categories].loc[(slice(None), "0.5Hz"), :].mean(axis=0).values[:-1], magn) 
OVR_10 = np.multiply(df_ovr[categories].loc[(slice(None), "1.0Hz"), :].mean(axis=0).values[:-1], magn) 
OVR_20 = np.multiply(df_ovr[categories].loc[(slice(None), "2.0Hz"), :].mean(axis=0).values[:-1], magn) 
    
OVR_01 = [*OVR_01, OVR_01[0]]
OVR_05 = [*OVR_05, OVR_05[0]]
OVR_10 = [*OVR_10, OVR_10[0]]
OVR_20 = [*OVR_20, OVR_20[0]]
    
    
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))

plt.figure(figsize=(12,12))
plt.subplot(polar=True)
plt.plot(label_loc, OVR_01, label='Overweighted - 0.1 Hz')
plt.plot(label_loc, OVR_05, label='Overweighted - 0.5 Hz')
plt.plot(label_loc, OVR_10, label='Overweighted - 1 Hz')
plt.plot(label_loc, OVR_20, label='Overweighted - 2 Hz')
plt.title('Average across participants', size=20, y=1.05)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
plt.legend()
plt.show()