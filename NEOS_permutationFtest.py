#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:47:15 2023

@author: fm02
"""
import numpy as np

from scipy import stats as stats

import mne

from mne.stats import (spatio_temporal_cluster_test, f_threshold_mway_rm,
                       f_mway_rm, summarize_clusters_stc)

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
import pickle

ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")

subjects_dir = config.subjects_dir

fname_fsaverage_src = path.join(subjects_dir,
                                'fsaverage',
                                'bem', 
                                'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(fname_fsaverage_src)

fsave_vertices = [src[0]['vertno'], []]
##################### very important to do
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# need to run from mne.epochs import equalize_epoch_counts
# in the epochs file which I have not done yet.

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

conditions = ['abspred', 'absunpred', 'concpred', 'concunpred'] 
stcs = dict()

for condition in conditions:
    stcs[condition]= []
    for sbj_id in sbj_ids:
        subject = str(sbj_id)
        # path to subject's data
        stc_fname = path.join(stc_path, 
                              f"{subject}_stc_{condition}_fsaverage")
    
        print('Reading source estimate from %s.' % stc_fname)
    
        stc = mne.read_source_estimate(stc_fname)
        stc.resample(250, npad='auto')
        stcs[condition].append(stc)
        # stcs[condition].append(stc.data)

cond1 = np.stack([stcs['abspred'][i].data for i in range(len(stcs['abspred']))])
cond2 = np.stack([stcs['absunpred'][i].data for i in range(len(stcs['absunpred']))])
cond3 = np.stack([stcs['concpred'][i].data for i in range(len(stcs['abspred']))])
cond4 = np.stack([stcs['concunpred'][i].data for i in range(len(stcs['concunpred']))])


X = np.stack([cond1, cond2, cond3, cond4], axis =-1)
#X = np.abs(X)

adjacency = mne.spatial_src_adjacency(src)

X = np.transpose(X, [0, 2, 1, 3])
X = [np.squeeze(x) for x in np.split(X, 4, axis=-1)]

# Here we set a cluster forming threshold based on a p-value for
# the cluster based permutation test.
# We use a two-tailed threshold, the "1 - p_threshold" is needed
# because for two-tailed tests we must specify a positive threshold.
p_threshold = 0.05
df = len(sbj_ids) - 1  # degrees of freedom for the test

# Now let's actually do the clustering. This can take a long time...
print('Clustering.')


factor_levels = [2, 2]
effects = 'A:B'
# Tell the ANOVA not to compute p-values which we don't need for clustering
return_pvals = False

# a few more convenient bindings
n_times = X[0].shape[1]
n_conditions = 4
n_subjects = len(sbj_ids)
n_permutations = 1024

def stat_fun(*args):
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=False)[0]

pthresh = 0.005
f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects, pthresh)

F_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_test(X, adjacency=adjacency, n_jobs=None,
                                 threshold=f_thresh, stat_fun=stat_fun,
                                 n_permutations=n_permutations,
                                 buffer_size=None)
    
with open('/imaging/hauk/users/fm02/MEG_NEOS/data/misc/interaction_noabs.P', 'wb') as f:
    pickle.dump(clu, f)

    