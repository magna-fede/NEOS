#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:46:41 2023

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

predictors = ['ConcCont', 'Length', 'Zipf', 'Position', 'PredCont']

rois_lab = ['lATL', 'rATL', 'PVA', 'IFG', 'AG', 'PTC']

all_predictors = dict().fromkeys(predictors)

for pred in all_predictors.keys():
    all_predictors[pred] = dict()
    for roi in rois_lab:
        all_predictors[pred][roi] = list()

        
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]


for sbj in sbj_ids:
    for pred in all_predictors.keys():
        with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/source_space/{sbj}_scores_{pred}.P', 'rb') as handle:
            c = pickle.load(handle)
        for roi in c.keys():
            all_predictors[pred][roi].append(np.array(c[roi].mean(axis=0)))
    
p_clust = dict()
for pred in all_predictors.keys():
    
    p_clust[pred] = pd.DataFrame(index=range(250), columns=rois_lab)
    for roi in all_predictors[pred].keys():
        # Reshape data to what is equivalent to (n_samples, n_space, n_time)
        data = np.array(all_predictors[pred][roi]).reshape(27, 1, 250)
        # Compute threshold from t distribution (this is also the default)
        threshold = stats.distributions.t.ppf(1 - 0.05, 27 - 1)
        t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
            data-.5, n_jobs=1, threshold=threshold, adjacency=None,
            n_permutations=5000)
        # Put the cluster data in a viewable format
        temp_p_clust = np.ones((1,250))
        for cl, p in zip(clusters, p_values):
            temp_p_clust[cl] = p
        p_clust[roi] = temp_p_clust.T
        
    for roi in all_predictors[pred].keys():
        fig, ax = plt.subplots(1)
        ax.plot(times, np.array(all_predictors[pred][roi]).mean(axis=0), label="score")
        ax.fill_between(x=times, \
                      y1=(np.array(all_predictors[pred][roi]).mean(axis=0) - np.array(all_predictors[pred][roi]).std(axis=0)), \
                      y2=(np.array(all_predictors[pred][roi]).mean(axis=0) + np.array(all_predictors[pred][roi]).std(axis=0)), \
                      alpha=.1)
        mask = np.array(p_clust[roi] < 0.05).reshape(250)
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
    
        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], alpha=0.5,
                        color="green")
        ax.axhline(0.5, color="k", linestyle="--", label="chance")
        ax.axvline(0, color="k")
        plt.title(f"{pred} - {roi}")
        plt.legend()
        plt.savefig(path.join(config.data_path, "plots", "decoding",
                              f"{pred}_{roi}_decoding_accuracy.png"))

# %%