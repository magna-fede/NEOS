#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 19:41:50 2023

@author: fm02
"""

from scipy import stats as stats

import pandas as pd
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sys
import os
from os import path
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

stcs_path = path.join(config.data_path, "stcs")

sbj_ids = [
            1,
            2,
            3,
        #   4, #fell asleep
            5,
            6,
        #    7, #no MRI
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
        #   20, #too magnetic to test
            21,
            22, 
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30
            ]

src_fname = path.join(config.subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif")

src = mne.read_source_spaces(src_fname)

predictables = [mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Predictable_eLORETA_MEGonly_auto_dropbads_fsaverage")) \
 for sbj_id in sbj_ids]


unpredictables = [mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Unpredictable_eLORETA_MEGonly_auto_dropbads_fsaverage")) \
 for sbj_id in sbj_ids]

data_p = [predictable.data.T for predictable in predictables]

data_u = [unpredictable.data.T for unpredictable in unpredictables]

X = np.stack(data_p) - np.stack(data_u)

adjacency = mne.spatial_src_adjacency(src)

p_threshold = 0.001
df = len(X) - 1  # degrees of freedom for the test
t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

clu = spatio_temporal_cluster_1samp_test(
    X,
    n_permutations=5000,
    adjacency=adjacency,
    threshold=t_threshold,
    buffer_size=None,
    verbose=True,
    n_jobs=-1
)

with open('/imaging/hauk/users/fm02/MEG_NEOS/data/misc/Predictability_MEGonly_unfold_Cluster1samp.P', 'wb') as f:
    pickle.dump(clu, f)
    
concretes = [mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Concrete_eLORETA_MEGonly_auto_dropbads_fsaverage")) \
 for sbj_id in sbj_ids]


abstracts = [mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Abstract_eLORETA_MEGonly_auto_dropbads_fsaverage")) \
 for sbj_id in sbj_ids]

data_c = [concrete.data.T for concrete in concretes]

data_a = [abstract.data.T for abstract in abstracts]

X = np.stack(data_c) - np.stack(data_a)

adjacency = mne.spatial_src_adjacency(src)

p_threshold = 0.001
df = len(X) - 1  # degrees of freedom for the test
t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

clu = spatio_temporal_cluster_1samp_test(
    X,
    n_permutations=5000,
    adjacency=adjacency,
    threshold=t_threshold,
    buffer_size=None,
    verbose=True,
    n_jobs=-1
)

with open('/imaging/hauk/users/fm02/MEG_NEOS/data/misc/Concreteness_MEGonly_unfold_Cluster1samp.P', 'wb') as f:
    pickle.dump(clu, f)
    

