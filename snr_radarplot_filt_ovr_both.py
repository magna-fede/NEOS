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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#mne.viz.set_browser_backend("matplotlib")

ovr_metrics = dict()
ovrons_metrics = dict()
novr_metrics = dict()

do_subjs = [
            1,
            2,
            3,
        #   4,
            5,
            6,
        #    7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18, 
            19, 
        #   20,
            21,
            22, 
            23, # check why overweighted onset did not work  
            24,
            25,
            26,
            27,
            28,
            29,
            30
            ]

for sbj_id in do_subjs:
    
    sbj_metrics_ovr = pd.DataFrame()
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    for blk in range(1, 6):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_compare_filt_{sbj_id}_{blk}_ovrw_both.csv")
                                  )
        sbj_metrics_ovr = pd.concat([sbj_metrics_ovr, blk_metrics], ignore_index=True)
    
    sbj_metrics_ovr = sbj_metrics_ovr.drop("Unnamed: 0", axis=1)
    sbj_metrics_ovr = sbj_metrics_ovr.groupby(["type"]).mean()
    
    ovr_metrics[sbj_id] = sbj_metrics_ovr
    
    sbj_metrics_novr = pd.DataFrame()
    
    for blk in range(1, 6):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_compare_filt_{sbj_id}_{blk}_both.csv")
                                  )
        sbj_metrics_novr = pd.concat([sbj_metrics_novr, blk_metrics], ignore_index=True)
    
    sbj_metrics_novr = sbj_metrics_novr.drop("Unnamed: 0", axis=1)
    sbj_metrics_novr = sbj_metrics_novr.groupby(["type"]).mean()
    novr_metrics[sbj_id] = sbj_metrics_novr
    
    sbj_metrics_ovrons = pd.DataFrame()
    
    for blk in range(1, 6):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_compare_filt_{sbj_id}_{blk}_ovrwonset_both.csv")
                                  )
        sbj_metrics_ovrons = pd.concat([sbj_metrics_ovrons, blk_metrics], ignore_index=True)
    
    sbj_metrics_ovrons = sbj_metrics_ovrons.drop("Unnamed: 0", axis=1)
    sbj_metrics_ovrons = sbj_metrics_ovrons.groupby(["type"]).mean()
    ovrons_metrics[sbj_id] = sbj_metrics_ovrons


df_ovr = pd.concat(ovr_metrics.values(), keys=ovr_metrics.keys())
df_ovrons = pd.concat(ovrons_metrics.values(), keys=ovrons_metrics.keys())
df_novr = pd.concat(novr_metrics.values(), keys=novr_metrics.keys())

norm_ovr = pd.DataFrame(columns=df_ovr.columns, index=df_ovr.index)
norm_ovrons = pd.DataFrame(columns=df_ovrons.columns, index=df_ovrons.index)
norm_novr = pd.DataFrame(columns=df_novr.columns, index=df_novr.index)

categories = [
              "P1_SNR",
              "GFP_first100",
              "GFP_baseline",
              "GFP_fixation_onset",
              "GFP_late",
              "S_amplitude",
              "S_auc",
             ]
for category in categories:
    scaler.fit(np.array(df_novr[category]).reshape(-1,1))
    norm_novr[category] = scaler.transform(np.array(df_novr[category]).reshape(-1,1))

    scaler.fit(np.array(df_ovr[category]).reshape(-1,1))
    norm_ovr[category] = scaler.transform(np.array(df_ovr[category]).reshape(-1,1))

    scaler.fit(np.array(df_ovrons[category]).reshape(-1,1))
    norm_ovrons[category] = scaler.transform(np.array(df_ovrons[category]).reshape(-1,1))

categories = [*categories, categories[0]]


for sbj_id in do_subjs:
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    OVR_01 = norm_ovr[categories].loc[sbj_id,"0.1Hz"].values[:-1]
    OVR_05 = norm_ovr[categories].loc[sbj_id,"0.5Hz"].values[:-1]
    OVR_10 = norm_ovr[categories].loc[sbj_id,"1.0Hz"].values[:-1]
    OVR_20 = norm_ovr[categories].loc[sbj_id,"2.0Hz"].values[:-1]

    OVR_01 = [*OVR_01, OVR_01[0]]
    OVR_05 = [*OVR_05, OVR_05[0]]
    OVR_10 = [*OVR_10, OVR_10[0]]
    OVR_20 = [*OVR_20, OVR_20[0]]
    
    nOVR_01 = norm_novr[categories].loc[sbj_id,"0.1Hz"].values[:-1]
    nOVR_05 = norm_novr[categories].loc[sbj_id,"0.5Hz"].values[:-1]
    nOVR_10 = norm_novr[categories].loc[sbj_id,"1.0Hz"].values[:-1]
    nOVR_20 = norm_novr[categories].loc[sbj_id,"2.0Hz"].values[:-1]

    nOVR_01 = [*nOVR_01, nOVR_01[0]]
    nOVR_05 = [*nOVR_05, nOVR_05[0]]
    nOVR_10 = [*nOVR_10, nOVR_10[0]]
    nOVR_20 = [*nOVR_20, nOVR_20[0]]
    
    OVRons_01 = norm_ovrons[categories].loc[sbj_id,"0.1Hz"].values[:-1]
    OVRons_05 = norm_ovrons[categories].loc[sbj_id,"0.5Hz"].values[:-1]
    OVRons_10 = norm_ovrons[categories].loc[sbj_id,"1.0Hz"].values[:-1]
    OVRons_20 = norm_ovrons[categories].loc[sbj_id,"2.0Hz"].values[:-1]

    OVRons_01 = [*OVRons_01, OVRons_01[0]]
    OVRons_05 = [*OVRons_05, OVRons_05[0]]
    OVRons_10 = [*OVRons_10, OVRons_10[0]]
    OVRons_20 = [*OVRons_20, OVRons_20[0]]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111, polar=True)    
    ax.plot(label_loc, OVR_01, label='Overweighted - 0.1 Hz')
    ax.plot(label_loc, OVR_05, label='Overweighted - 0.5 Hz')
    # ax.plot(label_loc, OVR_10, label='Overweighted - 1.0 Hz')
    # ax.plot(label_loc, OVR_20, label='Overweighted - 2.0 Hz')
    
    ax.plot(label_loc, OVRons_01, label='Overweighted onset - 0.1 Hz')
    ax.plot(label_loc, OVRons_05, label='Overweighted onset - 0.5 Hz')
    # ax.plot(label_loc, OVRons_10, label='Overweighted onset - 1.0 Hz')
    # ax.plot(label_loc, OVRons_20, label='Overweighted onset - 2.0 Hz')
    
    ax.plot(label_loc, nOVR_01, label='Non-overweighted - 0.1 Hz')
    ax.plot(label_loc, nOVR_05, label='Non-overweighted - 0.5 Hz')
    # ax.plot(label_loc, nOVR_10, label='Non-overweighted - 1.0 Hz')
    # ax.plot(label_loc, nOVR_20, label='Non-overweighted - 2.0 Hz')
    
    ax.fill(label_loc, OVR_01, label='Overweighted - 0.1 Hz', alpha=0.1)
    # plt.title(f'Participant {sbj_id}', size=20, y=1.05)

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), ncol=1)
    leg = leg.set_in_layout(True)
    fig.tight_layout()
    fname_fig = path.join(sbj_path, 'Figures', f'snr_EEG_filtering.png')
    fig.savefig(fname_fig)

    #plt.show()


OVR_01Hz = norm_ovr[categories].loc[(slice(None), "0.1Hz"), :].mean(axis=0) 
OVR_05Hz = norm_ovr[categories].loc[(slice(None), "0.5Hz"), :].mean(axis=0) 
OVR_10Hz = norm_ovr[categories].loc[(slice(None), "1.0Hz"), :].mean(axis=0) 
OVR_20Hz = norm_ovr[categories].loc[(slice(None), "2.0Hz"), :].mean(axis=0) 

OVRons_01Hz = norm_ovrons[categories].loc[(slice(None), "0.1Hz"), :].mean(axis=0) 
OVRons_05Hz = norm_ovrons[categories].loc[(slice(None), "0.5Hz"), :].mean(axis=0) 
OVRons_10Hz = norm_ovrons[categories].loc[(slice(None), "1.0Hz"), :].mean(axis=0) 
OVRons_20Hz = norm_ovrons[categories].loc[(slice(None), "2.0Hz"), :].mean(axis=0) 

nOVR_01Hz = norm_novr[categories].loc[(slice(None), "0.1Hz"), :].mean(axis=0) 
nOVR_05Hz = norm_novr[categories].loc[(slice(None), "0.5Hz"), :].mean(axis=0) 
nOVR_10Hz = norm_novr[categories].loc[(slice(None), "1.0Hz"), :].mean(axis=0)
nOVR_20Hz = norm_novr[categories].loc[(slice(None), "2.0Hz"), :].mean(axis=0) 

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))

fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111, polar=True)    
ax.plot(label_loc, OVR_01Hz, label='Overweighted - 0.1Hz')
ax.plot(label_loc, OVR_05Hz, label='Overweighted - 0.5Hz')

ax.plot(label_loc, OVRons_01Hz, label='Overweighted onset - 0.1Hz')
ax.plot(label_loc, OVRons_05Hz, label='Overweighted onset - 0.5Hz')

ax.plot(label_loc, nOVR_01Hz, label='Non-overweighted - 0.1Hz')
ax.plot(label_loc, nOVR_05Hz, label='Non-overweighted - 0.5Hz')

ax.fill(label_loc, OVR_01Hz, label='Overweighted - 0.1Hz', alpha=0.1)
# plt.title('Average across participants', size=20, y=1.05)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
leg = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), ncol=1)
leg = leg.set_in_layout(True)
fig.tight_layout()
fname_fig = path.join(config.data_path, 'misc', f'avg_snr_EEG_01_01Hz.png')
fig.savefig(fname_fig)

plt.show()

summary_ovr = df_ovr[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["0.1Hz", "2.0Hz", "0.5Hz", "1.0Hz"]), :]
summary_ovrons = df_ovrons[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["0.1Hz", "2.0Hz", "0.5Hz", "1.0Hz"]), :]
summary_novr = df_novr[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["0.1Hz", "2.0Hz", "0.5Hz", "1.0Hz"]), :]


print("No overweighting: \n", summary_novr.groupby(["type"]).mean(), "\n")
print("Saccade overweighting: \n",summary_ovr.groupby(["type"]).mean(), "\n")
print("Saccade onset overweighting: \n",summary_ovrons.groupby(["type"]).mean(), "\n")