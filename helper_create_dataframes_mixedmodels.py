#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:58:47 2023

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

rois_lab = ['lATL', 'rATL', 'PVA', 'IFG', 'AG', 'PTC', 'avg']

all_predictors = dict().fromkeys(predictors)

for pred in all_predictors.keys():
    all_predictors[pred] = dict()
    for roi in rois_lab:
        all_predictors[pred][roi] = list()

        
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

# %% Create models for factorial
all_subjs = list()
for subject in sbj_ids:
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/data_for_mixed_models/sbj_{subject}_factorial.P', 'rb') as handle:
        all_subjs.append(pickle.load(handle))
        
times = all_subjs[0].keys()

for t in times:
    one_time = pd.concat([one_sbj[t] for one_sbj in all_subjs])
    for roi in rois_lab:
        inroi = one_time[one_time['roi']==roi]
        inroi.to_csv(f'/imaging/hauk/users/fm02/MEG_NEOS/data/data_for_mixed_models/{roi}_{t}_factorial.csv', index=False)
        
        
        
        
        
        
        