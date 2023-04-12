#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:47:15 2023

@author: fm02
"""
import numpy as np

from scipy import stats as stats

import mne

from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
import pickle

ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")

subjects_dir = config.subjects_dir

stc_sub='eLORETA'

fname_fsaverage_src = path.join(subjects_dir,
                                'fsaverage',
                                'bem', 
                                'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(fname_fsaverage_src)

fsave_vertices = [src[0]['vertno'], src[1]['vertno']]
##################### very important to do
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# need to run from mne.epochs import equalize_epoch_counts
# in the epochs file which I have not done yet.

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

factors = dict(Predictability=['Predictable', 'Unpredictable'], Concreteness=['Concrete', 'Abstract'])

for factor, conditions in factors.items():
        
    stcs = dict()
    
    for condition in conditions:
        stcs[condition]= []
        for sbj_id in sbj_ids:
            subject = str(sbj_id)
            # path to subject's data
            stc_fname = path.join(stc_path, 
                                  f"{subject}_stc_{condition}_{stc_sub}_fsaverage")
        
            print('Reading source estimate from %s.' % stc_fname)
        
            stc = mne.read_source_estimate(stc_fname)
            stc.resample(250, npad='auto')
            stcs[condition].append(stc)
            # stcs[condition].append(stc.data)
    
    cond1 = np.stack([stcs[conditions[0]][i].data for i in range(len(stcs[conditions[0]]))])
    cond2 = np.stack([stcs[conditions[1]][i].data for i in range(len(stcs[conditions[1]]))])
    
    # cond1 = np.stack([stcs['Concrete'][i].data for i in range(len(stcs['Concrete']))])
    # cond2 = np.stack([stcs['Abstract'][i].data for i in range(len(stcs['Abstract']))])
    
    X = np.stack([cond1,cond2], axis =-1)
    X = np.abs(X)
    X = X[:, :, :, 0] - X[:, :, :, 1]
    
    adjacency = mne.spatial_src_adjacency(src)
    
    X = np.transpose(X, [0, 2, 1])
    
    # Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed
    # because for two-tailed tests we must specify a positive threshold.
    p_threshold = 0.05
    df = len(sbj_ids) - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    
    # Now let's actually do the clustering. This can take a long time...
    print('Clustering.')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, adjacency=adjacency,
                                           threshold=t_threshold, buffer_size=None,
                                           verbose=True, n_jobs=-1)
        
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/misc/cluster1samp_{factor}{stc_sub}.P', 'wb') as f:
        pickle.dump(clu, f)
        
# with open('/imaging/hauk/users/fm02/MEG_NEOS/data/misc/cluster1sampPredictability_normalorientation.P', 'rb') as f:
#     pickle.load(f)
    
# with open('/imaging/hauk/users/fm02/MEG_NEOS/data/misc/cluster1sampConcreteness_normalorientation.P', 'rb') as f:
#     clue= pickle.load(f)
    
# good_clusters_idx = np.where(clue[2] < 0.05)[0]
# good_clusters = [clue[1][idx] for idx in good_clusters_idx]

# stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=1/250, tmin=-0.3,
#                                              vertices=fsave_vertices,
#                                              subject='fsaverage')

# # Let's actually plot the first "time point" in the SourceEstimate, which
# # shows all the clusters, weighted by duration.

# # blue blobs are for condition A < condition B, red for A > B
# brain = stc_all_cluster_vis.plot(
#     hemi='both', views='lateral', subjects_dir=config.subjects_dir,
#     time_label='temporal extent (ms)', size=(800, 800),
#     smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]))

