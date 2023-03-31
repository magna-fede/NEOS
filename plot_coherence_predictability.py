#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:21:26 2023

@author: fm02
"""

import sys
import os
from os import path

import numpy as np
import pandas as pd

import mne

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from mne.viz import circular_layout
from mne_connectivity import (read_connectivity, spectral_connectivity_epochs,
                              )
from mne_connectivity.viz import plot_connectivity_circle

import matplotlib.pyplot as plt
import seaborn as sns

labels_path = path.join(config.data_path, "my_ROIs")

predictability_factors = ['Predictable', 'Unpredictable']
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

overall = dict()
for condition in predictability_factors:
    overall[condition] = []
    
for sbj_id in sbj_ids:
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    for condition in overall.keys():
        overall[condition].append(read_connectivity(path.join(sbj_path,
                                        f"{sbj_id}_{condition}_ROI_coherence")
                                                    )
                                  )
alpha = []
beta = []
for coh in overall['Unpredictable']:
    alpha.append(coh.get_data()[:, 0])
    beta.append(coh.get_data()[:, 1])
    
GA_alpha = np.stack(alpha).mean(axis=0)
GA_beta = np.stack(beta).mean(axis=0)

GA_alpha = GA_alpha.reshape(6,6)
GA_beta = GA_beta.reshape(6,6)

label_names=['l_ATL', 'r_ATL', 'PTC', 'IFG', 'AG', 'PVA']
lATL = mne.read_label(path.join(labels_path, 'l_ATL_fsaverage-lh.label'),
                      subject='fsaverage')
rATL = mne.read_label(path.join(labels_path, 'r_ATL_fsaverage-rh.label'),
                      subject='fsaverage')
PVA = mne.read_label(path.join(labels_path, 'PVA_fsaverage-lh.label'),
                      subject='fsaverage')
IFG = mne.read_label(path.join(labels_path, 'IFG_fsaverage-lh.label'),
                      subject='fsaverage')
AG = mne.read_label(path.join(labels_path, 'AG_fsaverage-lh.label'),
                      subject='fsaverage')
PTC = mne.read_label(path.join(labels_path, 'PTC_fsaverage-lh.label'),
                      subject='fsaverage')


rois = [lATL,
        rATL, 
        PVA,
        IFG,
        AG,
        PTC]
label_colors = sns.color_palette(['#FFBE0B',
                            '#FB5607',
                            '#FF006E',
                            '#8338EC',
                            '#3A86FF',
                            '#1D437F',
                            '#1D437F'
                            ])
label_ypos = list()
for name in label_names:
    idx = label_names.index(name)
    ypos = np.mean(rois[idx].pos[:, 1])
    label_ypos.append(ypos)
    
node_order = ['l_ATL', 'r_ATL', 'PTC', 'IFG', 'AG', 'PVA']
node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
fig = plot_connectivity_circle(GA_alpha, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='All-to-All Connectivity Alpha range '
                               'Coherence', ax=ax)
plt.savefig(path.join(config.data_path, 'misc',
                      'grandaverageCoherence_alpha.png'), format='png');

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
plot_connectivity_circle(GA_beta, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='All-to-All Connectivity Beta range '
                               'Coherence', ax=ax)
plt.savefig(path.join(config.data_path, 'misc',
                      'grandaverageCoherence_beta.png'), format='png');

