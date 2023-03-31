#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:15:48 2023

@author: fm02
"""

import numpy as np
import matplotlib.pyplot as plt

import mne

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mne.stats import permutation_cluster_1samp_test
from scipy import stats

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

stc_rois = dict()
avgs = dict()

for roi in rois_lab:
    stc_rois[roi] = [] 
    avgs[roi] = []
    
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

for sbj in sbj_ids:
    
    stc = mne.read_source_estimate(path.join(stc_path, f'{sbj}_stc_predictable_fsaverage'))
    
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
    
    # for roi in rois:
    #     stc_rois[roi.name].append(stc.in_label(roi))
    #     avgs[roi.name].append(stc.extract_label_time_course(roi, src, mode='mean'))        
    

    # for roi in rois:

    #     fig, ax = plt.subplots(1);
    #     for i in range(0, 27):
    #         ax.plot(times, avgs[roi.name][i].T, 'k', linewidth=0.1, alpha=0.5);
    #     ax.plot(times, np.concatenate(avgs[roi.name]).mean(axis=0), linewidth=2)
    #     ax.set(xlabel='Time (ms)', ylabel='Source amplitude',
    #        title='Activations in Label %r' % (roi.name))
    #     plt.show()

stc_rois_unp = dict()
avgs_unp = dict()

for roi in rois_lab:
    stc_rois_unp[roi] = [] 
    avgs_unp[roi] = []
    
for sbj in sbj_ids:
    
    stc = mne.read_source_estimate(path.join(stc_path, f'{sbj}_stc_unpredictable_fsaverage'))
    
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
        stc_rois_unp[roi.name].append(stc.in_label(roi))
        avgs_unp[roi.name].append(stc.extract_label_time_course(roi, src, mode='mean'))        
    
test_pred = dict()
test_unpred = dict()
for roi in rois_lab:
    test_pred[roi] = np.concatenate(avgs[roi])
    test_unpred[roi] = np.concatenate(avgs_unp[roi])
    
predictability_effect = dict()    
for roi in rois_lab:
    predictability_effect[roi] = test_pred[roi] - test_unpred[roi]


p_clust = {}
t_clust = {}
clusters = {}
p_values = {}
H0 = {} 
p_clust

for roi in rois_lab:
    p_clust[roi] = pd.DataFrame(index=range(1001))
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    score = np.stack(predictability_effect[roi]).reshape(27,1,1001)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 27 - 1)
    t_clust[roi], clusters[roi], p_values[roi], H0[roi] = permutation_cluster_1samp_test(
        score-.5, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations='all')
    # Put the cluster data in a viewable format
    temp_p_clust = np.ones((1,300))
    for cl, p in zip(clusters[roi], p_values[roi]):
        temp_p_clust[cl] = p
    p_clust[roi] = temp_p_clust.T
  
    
