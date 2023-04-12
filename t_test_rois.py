#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:33:56 2023

@author: fm02
"""
import numpy as np
import matplotlib.pyplot as plt

import mne

from mne.datasets import sample

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

stc_path = path.join(config.data_path, "stcs")
subjects_dir = config.subjects_dir
labels_dir = path.join(config.data_path, "my_ROIs")
labels_path = path.join(config.data_path, "my_ROIs")

fname_fsaverage_src = path.join(subjects_dir,
                                'fsaverage',
                                'bem', 
                                'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(fname_fsaverage_src)

times=np.arange(-300,701,1)

rois_lab = ['lATL',
            'rATL', 
            'PVA',
            'IFG',
            'AG',
            'PTC']

unpred_eloreta = dict()
pred_eloreta = dict()

for roi in rois_lab:
    unpred_eloreta[roi] = [] 
    pred_eloreta[roi] = []

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]
    
for sbj in sbj_ids:
    
    stc = mne.read_source_estimate(path.join(stc_path, f'{sbj}_stc_Concrete_eLORETA_fsaverage'))
    
    lATL = mne.read_label(path.join(labels_path, 'l_ATL_fsaverage-lh.label'),
                          subject='fsaverage')
    lATL.name='lATL'
    rATL = mne.read_label(path.join(labels_path, 'r_ATL_fsaverage-rh.label'),
                          subject='fsaverage')
    rATL.name='rATL'
    PVA = mne.read_label(path.join(labels_path, 'PVA_fsaverage-lh.label'),
                          subject='fsaverage')
    PVA.name='PVA'
    IFG = mne.read_label(path.join(labels_path, 'IFG_fsaverage-lh.label'),
                          subject='fsaverage')
    IFG.name='IFG'
    AG = mne.read_label(path.join(labels_path, 'AG_fsaverage-lh.label'),
                          subject='fsaverage')
    AG.name='AG'
    PTC = mne.read_label(path.join(labels_path, 'PTC_fsaverage-lh.label'),
                          subject='fsaverage')
    PTC.name='PTC'

    rois = [lATL,
            rATL, 
            PVA,
            IFG,
            AG,
            PTC]
    
    for roi in rois:
        pred_eloreta[roi.name].append(stc.extract_label_time_course(roi, src, mode='mean'))        


for sbj in sbj_ids:
    
    stc = mne.read_source_estimate(path.join(stc_path, f'{sbj}_stc_Abstract_eLORETA_fsaverage'))
    
    lATL = mne.read_label(path.join(labels_path, 'l_ATL_fsaverage-lh.label'),
                          subject='fsaverage')
    lATL.name='lATL'
    rATL = mne.read_label(path.join(labels_path, 'r_ATL_fsaverage-rh.label'),
                          subject='fsaverage')
    rATL.name='rATL'
    PVA = mne.read_label(path.join(labels_path, 'PVA_fsaverage-lh.label'),
                          subject='fsaverage')
    PVA.name='PVA'
    IFG = mne.read_label(path.join(labels_path, 'IFG_fsaverage-lh.label'),
                          subject='fsaverage')
    IFG.name='IFG'
    AG = mne.read_label(path.join(labels_path, 'AG_fsaverage-lh.label'),
                          subject='fsaverage')
    AG.name='AG'
    PTC = mne.read_label(path.join(labels_path, 'PTC_fsaverage-lh.label'),
                          subject='fsaverage')
    PTC.name='PTC'

    rois = [lATL,
            rATL, 
            PVA,
            IFG,
            AG,
            PTC]
    
    for roi in rois:
        unpred_eloreta[roi.name].append(stc.extract_label_time_course(roi, src, mode='mean'))        

from scipy import stats
from mne.stats import permutation_cluster_test

threshold_uncorrected = stats.t.ppf(1.0 - 0.05, 27 - 1)

results = dict()
for roi in rois_lab:
    diff = np.array(pred_eloreta[roi])-np.array(unpred_eloreta[roi])
    results[roi] = stats.ttest_1samp(diff, popmean=0)

T_obs = dict()
clusters = dict()
cluster_p_values = dict()
H0 = dict()

threshold=6.0
for roi in rois_lab:
    T_obs[roi], clusters[roi], cluster_p_values[roi], H0[roi] = \
    permutation_cluster_test([np.array(pred_eloreta[roi]), np.array(unpred_eloreta[roi])], n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=None,
                             out_type='mask')

for roi in rois_lab:
    times=stc.times
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i_c, c in enumerate(clusters[roi]):
        c = c[0]
        # if cluster_p_values[roi][i_c] <= 0.05:
        #     h = ax.axvspan(times[c.start], times[c.stop - 1],
        #                     color='r', alpha=0.3)
        # else:
        #     ax.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
        #                 alpha=0.3)
    
    fig = plt.plot(times, T_obs[roi].squeeze(), 'g')
#    ax.legend((h, ), ('cluster p-value < 0.05', ))
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("f-values")
    ax.set_title(roi)

for roi in rois_lab:
    fig, ax = plt.subplots(1)
    fig = sns.lineplot(x=times,y=results[roi].statistic.squeeze())
    fig = plt.axhline(y = threshold_uncorrected, color = 'r', linestyle = '-')
    fig = plt.axhline(y = -threshold_uncorrected, color = 'r', linestyle = '-')
    ax.set_title(roi)
    plt.show()
