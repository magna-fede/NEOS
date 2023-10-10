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

import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

times = np.arange(-0.200, 0.500, 0.004)

rois_lab = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
concs = dict()
preds = dict()

for roi in rois_lab:
    concs[roi] = list()
    preds[roi] = list()
    
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

all_predictors = dict().fromkeys(['Concreteness', 'Predictability'])

all_predictors['Concreteness'] = concs
all_predictors['Predictability'] = preds


for sbj in sbj_ids:
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/source_space/{sbj}_scores_3pseudotrials_source.P', 'rb') as handle:
        c = pickle.load(handle)
    for roi in rois_lab:
        for pred in all_predictors.keys():
            all_predictors[pred][roi].append(np.array(c[roi][pred].mean(axis=0)))

len_times = len(all_predictors[pred]['lATL'][0])

p_clust = dict().fromkeys(['Predictability', 'Concreteness'])

for key in p_clust.keys():
    p_clust[key] = pd.DataFrame(index=range(len_times), columns=rois_lab)

for task in ['Predictability', 'Concreteness']:

    for roi in rois_lab:
        # Reshape data to what is equivalent to (n_samples, n_space, n_time)
        data = np.array(all_predictors[task][roi]).reshape(27, 1, len_times)
        # Compute threshold from t distribution (this is also the default)
        threshold = stats.distributions.t.ppf(1 - 0.05, 27 - 1)
        t_clust, clusters, p_values, H0 = permutation_cluster_1samp_test(
            data-.5, n_jobs=1, threshold=threshold, adjacency=None,
            n_permutations=5000)
        # Put the cluster data in a viewable format
        temp_p_clust = np.ones((1, len_times))
        for cl, p in zip(clusters, p_values):
            temp_p_clust[cl] = p
        p_clust[task][roi] = temp_p_clust.T

colors = sns.color_palette([
                            '#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F',
                            ])

for task in p_clust.keys():
    for roi in p_clust[task].columns:
        print(f"{task, roi}: Decoding [task] at timepoints: \
              {times[np.where(p_clust[task][roi] < 0.1)[0]]}")
        #scores[task][roi].shape = (18, 300)


for task in all_predictors.keys():    
    for i, roi in enumerate(all_predictors[task].keys()):
        fig, ax = plt.subplots(1)
        ax.plot(times, np.array(all_predictors[task][roi]).mean(axis=0), color=colors[i])
        ax.fill_between(x=times, \
                      y1=(np.array(all_predictors[task][roi]).mean(axis=0) - sem(np.array(all_predictors[task][roi]), 0)), \
                      y2=(np.array(all_predictors[task][roi]).mean(axis=0) + sem(np.array(all_predictors[task][roi]), 0)), \
                      alpha=.1, color=colors[i])
        ax.axhline(0.5, color="k", linestyle="--", label="chance")
        ax.axvline(0, color="k")

        mask = p_clust[task][roi] < 0.05
        mask = mask.values.reshape(len_times)
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], alpha=0.5,
                        label="Cluster based permutation p<.05",
                        color="green")
            
        mask = p_clust[task][roi] < 0.1
        mask = mask.values.reshape(len_times)
        first_vals = np.argwhere((~mask[:-1] & mask[1:]))  # Look for False-True transitions
        last_vals = np.argwhere((mask[:-1] & ~mask[1:])) + 1  # Look for True-False transitions
        for start, stop in zip(first_vals, last_vals):
            plt.axvspan(times[start], times[stop], alpha=0.3,
                        label="Cluster based permutation p<.1",
                        color="yellow")
        

        plt.title(f"{roi}", fontsize="20")
        plt.ylim([0.44, 0.58])
        leg = plt.legend()
        ax.get_legend().set_visible(False) 
        plt.tight_layout()
        plt.savefig(path.join(config.data_path, "misc", 
                              f"{task}_{roi}_source_decoding_accuracy.png"))