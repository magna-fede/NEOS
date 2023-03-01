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
    
    for i in range(5):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_componentselection_{sbj_id}_{i+1}_overweight.csv")
                                  )
        sbj_metrics_ovr = pd.concat([sbj_metrics_ovr, blk_metrics], ignore_index=True)
    
    sbj_metrics_ovr = sbj_metrics_ovr.drop("Unnamed: 0", axis=1)
    sbj_metrics_ovr = sbj_metrics_ovr.groupby(["type"]).mean()
    
    ovr_metrics[sbj_id] = sbj_metrics_ovr

    sbj_metrics_ovrons = pd.DataFrame()
    
    for i in range(5):
    
        blk_metrics = pd.read_csv(path.join(sbj_path,
                                            f"snr_componentselection_{sbj_id}_{i+1}_overweight_onset.csv")
                                  )
        sbj_metrics_ovrons = pd.concat([sbj_metrics_ovrons, blk_metrics], ignore_index=True)
    
    sbj_metrics_ovrons = sbj_metrics_ovrons.drop("Unnamed: 0", axis=1)
    sbj_metrics_ovrons = sbj_metrics_ovrons.groupby(["type"]).mean()
    
    ovrons_metrics[sbj_id] = sbj_metrics_ovrons
    
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
df_ovrons = pd.concat(ovrons_metrics.values(), keys=ovrons_metrics.keys())


categories = ['P1_SNR', 'GFP_first100', 'SNR_n400','S_amplitude','S_auc']

# SNR does not need to be normalised as it has not unit, but other values 
# should be somehow scaled for comparison

norm_ovr = pd.DataFrame(columns=df_ovr.columns, index=df_ovr.index)
norm_ovrons = pd.DataFrame(columns=df_ovrons.columns, index=df_ovrons.index)
norm_novr = pd.DataFrame(columns=df_novr.columns, index=df_novr.index)

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
    
    OVR_both = norm_ovr[categories].loc[sbj_id,"both"].values[:-1]
    OVR_eog = norm_ovr[categories].loc[sbj_id,"eog"].values[:-1]
    OVR_preICA = norm_ovr[categories].loc[sbj_id,"pre-ICA"].values[:-1]
    OVR_var = norm_ovr[categories].loc[sbj_id,"variance"].values[:-1]
    
    OVRons_both = norm_ovrons[categories].loc[sbj_id,"both"].values[:-1]
    OVRons_eog = norm_ovrons[categories].loc[sbj_id,"eog"].values[:-1]
    OVRons_preICA = norm_ovrons[categories].loc[sbj_id,"pre-ICA"].values[:-1]
    OVRons_var = norm_ovrons[categories].loc[sbj_id,"variance"].values[:-1]
       
    nOVR_both = norm_novr[categories].loc[sbj_id,"both"].values[:-1]
    nOVR_eog = norm_novr[categories].loc[sbj_id,"eog"].values[:-1]
#    nOVR_preICA = norm_novr[categories].loc[sbj_id,"pre-ICA"].values[:-1]
    nOVR_var = norm_novr[categories].loc[sbj_id,"variance"].values[:-1]
    
    
    OVR_both = [*OVR_both, OVR_both[0]]
    OVR_eog = [*OVR_eog, OVR_eog[0]]
    OVR_preICA = [*OVR_preICA, OVR_preICA[0]]
    OVR_var = [*OVR_var, OVR_var[0]]
 
    OVRons_both = [*OVRons_both, OVRons_both[0]]
    OVRons_eog = [*OVRons_eog, OVRons_eog[0]]
    OVRons_preICA = [*OVRons_preICA, OVRons_preICA[0]]
    OVRons_var = [*OVRons_var, OVRons_var[0]]
    
    nOVR_both = [*nOVR_both, nOVR_both[0]]
    nOVR_eog = [*nOVR_eog, nOVR_eog[0]]
#    nOVR_preICA = [*nOVR_preICA, nOVR_preICA[0]]
    nOVR_var = [*nOVR_var, nOVR_var[0]]
    
    
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
    
    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot(111, polar=True)    
    ax.plot(label_loc, OVR_both, label='Overweighted - Both')
    #ax.plot(label_loc, OVR_eog, label='Overweighted - EOG')
    #ax.plot(label_loc, OVR_preICA, label='Overweighted - preICA')
    # ax.plot(label_loc, OVR_var, label='Overweighted - Variance')
    ax.plot(label_loc, OVRons_both, label='Overweighted onset - Both')
    # ax.plot(label_loc, OVRons_eog, label='Overweighted onset - EOG')
    # ax.plot(label_loc, OVRons_preICA, label='Overweighted onset - preICA')
    # ax.plot(label_loc, OVRons_var, label='Overweighted onset - Variance')
    ax.plot(label_loc, nOVR_both, label='Non-overweighted - Both')
    # ax.plot(label_loc, nOVR_eog, label='Non-overweighted - EOG')
#    plt.plot(label_loc, nOVR_preICA, label='Non-overweighted - preICA')
    # ax.plot(label_loc, nOVR_var, label='Non-overweighted - Variance')
    ax.fill(label_loc, OVR_both, label='Overweighted - Both', alpha=0.1)
    #plt.title(f'Participant {sbj_id}', size=20, y=1.05)
    
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), ncol=1)
    leg = leg.set_in_layout(True)
    fig.tight_layout()
    fname_fig = path.join(sbj_path, 'Figures',  f'snr_EEG_all_01Hz.png')
    fig.savefig(fname_fig)
    #plt.show()
    

# average SNR

OVR_both = norm_ovr[categories].loc[(slice(None), "both"), :].mean(axis=0) 
OVR_eog = norm_ovr[categories].loc[(slice(None), "eog"), :].mean(axis=0) 
OVR_preICA = norm_ovr[categories].loc[(slice(None), "pre-ICA"), :].mean(axis=0) 
OVR_var = norm_ovr[categories].loc[(slice(None), "variance"), :].mean(axis=0) 

OVRons_both = norm_ovrons[categories].loc[(slice(None), "both"), :].mean(axis=0) 
OVRons_eog = norm_ovrons[categories].loc[(slice(None), "eog"), :].mean(axis=0) 
OVRons_preICA = norm_ovrons[categories].loc[(slice(None), "pre-ICA"), :].mean(axis=0) 
OVRons_var = norm_ovrons[categories].loc[(slice(None), "variance"), :].mean(axis=0) 

nOVR_both = norm_novr[categories].loc[(slice(None), "both"), :].mean(axis=0) 
nOVR_eog = norm_novr[categories].loc[(slice(None), "eog"), :].mean(axis=0) 
#    nOVR_preICA = norm_novr[categories].loc[(slice(None), "pre-ICA"), :].mean(axis=0)
nOVR_var = norm_novr[categories].loc[(slice(None), "variance"), :].mean(axis=0) 

label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))

fig = plt.figure(figsize=(20,10))
ax = plt.subplot(111, polar=True)    
ax.plot(label_loc, OVR_both, label='Overweighted - Both')
#ax.plot(label_loc, OVR_eog, label='Overweighted - EOG')
#ax.plot(label_loc, OVR_preICA, label='preICA')
#ax.plot(label_loc, OVRons_var, label='Overweighted onset - Variance')
ax.plot(label_loc, OVRons_both, label='Overweighted onset - Both')
#ax.plot(label_loc, OVRons_eog, label='Overweighted onset - EOG')
#ax.plot(label_loc, OVRons_var, label='Overweighted - Variance')
ax.plot(label_loc, nOVR_both, label='Non-overweighted - Both')
#ax.plot(label_loc, nOVR_eog, label='Non-overweighted - EOG')
#    plt.plot(label_loc, nOVR_preICA, label='Non-overweighted - preICA')
#ax.plot(label_loc, nOVR_var, label='Non-overweighted - Variance')
ax.fill(label_loc, OVR_both, label='Overweighted - Both', alpha=0.1)
#plt.title('Average across participants', size=20, y=1.05)

lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
leg = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 0.5), ncol=1)
leg = leg.set_in_layout(True)
fig.tight_layout()
fname_fig = path.join(config.data_path, 'misc', f'avg_snr_EEG_both_01Hz.png')
fig.savefig(fname_fig)
#plt.show()
    

summary_ovr = df_ovr[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["both", "variance", "eog", "pre-ICA"]), :]
summary_ovrons = df_ovrons[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["both", "variance", "eog", "pre-ICA"]), :]
summary_novr = df_novr[['P1_SNR', 'SNR_n400','S_amplitude','S_auc']].loc[(slice(None), ["both", "variance", "eog", "pre-ICA"]), :]


print("No overweighting: \n", summary_novr.groupby(["type"]).mean(), "\n")
print("Saccade overweighting: \n",summary_ovr.groupby(["type"]).mean(), "\n")
print("Saccade onset overweighting: \n",summary_ovrons.groupby(["type"]).mean(), "\n")