#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:00:58 2023

@author: fm02
"""

import os
from os import path
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from scipy import stats
from mne.stats import permutation_cluster_1samp_test

os.chdir("/home/fm02/MEG_NEOS/NEOS")

import NEOS_config as config
times = np.linspace(-300, 696, 250)

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]


all_predictors = dict().fromkeys(['Concreteness', 'Predictability'])

for pred in all_predictors.keys():
    all_predictors[pred] = list()


for sbj in sbj_ids:
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/source_space/{sbj}_scores_sensor.P', 'rb') as handle:
        c = pickle.load(handle)
    for pred in all_predictors.keys():
        all_predictors[pred].append(np.array(c[pred].mean(axis=0)))

p_clust = dict()
for pred in all_predictors.keys():
    
    p_clust[pred] = pd.DataFrame(index=range(250), columns=rois_lab)
    # Reshape data to what is equivalent to (n_samples, n_space, n_time)
    data = np.array(all_predictors[pred]).reshape(27, 1, 250)
    # Compute threshold from t distribution (this is also the default)
    threshold = stats.distributions.t.ppf(1 - 0.05, 27 - 1)
    t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
        data-.5, n_jobs=1, threshold=threshold, adjacency=None,
        n_permutations=5000)
    # Put the cluster data in a viewable format
    temp_p_clust = np.ones((1,250))
    for cl, p in zip(clusters, p_values):
        temp_p_clust[cl] = p
    p_clust[pred] = temp_p_clust.T
        

    fig, ax = plt.subplots(1)
    ax.plot(times, np.array(all_predictors[pred]).mean(axis=0), label="score")
    ax.fill_between(x=times, \
                  y1=(np.array(all_predictors[pred]).mean(axis=0) - np.array(all_predictors[pred]).std(axis=0)), \
                  y2=(np.array(all_predictors[pred]).mean(axis=0) + np.array(all_predictors[pred]).std(axis=0)), \
                  alpha=.1)
    mask = np.array(p_clust[pred] < 0.05).reshape(250)
    first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
    last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions

    for start, stop in zip(first_vals, last_vals):
        plt.axvspan(times[start], times[stop], alpha=0.5,
                    color="green")
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.axvline(0, color="k")
    plt.title(f"{pred}")
    plt.legend()
        plt.savefig(path.join(config.data_path, "plots", "decoding",
                              f"{pred}_{roi}_decoding_accuracy.png"))