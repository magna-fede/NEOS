#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:00:13 2023

@author: fm02
"""

import numpy as np
import matplotlib.pyplot as plt

import mne

from mne.minimum_norm import (read_inverse_operator,
                              make_inverse_resolution_matrix,
                              get_point_spread)

from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

overall = list()

for sbj_id in sbj_ids:
    subject = str(sbj_id)
    subjects_dir = config.subjects_dir
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0]) 
    
    fwd_fname = path.join(sbj_path, subject + '_EEGMEG-fwd.fif')
    
    inv_fname = path.join(sbj_path, subject + '_EEGMEG-inv_emp3150.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    forward = mne.read_forward_solution(fwd_fname)
    mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, copy=False)
    rm_mne = make_inverse_resolution_matrix(forward, inverse_operator,
                                            method='MNE', lambda2=1. / 3.**2)
    src = inverse_operator['src']
    del forward, inverse_operator  # save memory
    labels_path = path.join(config.data_path, "my_ROIs")
    
    lATL = mne.read_label(path.join(labels_path, 'l_ATL_fsaverage-lh.label'),
                          subject=subject)
    lATL.name='lATL'
    rATL = mne.read_label(path.join(labels_path, 'r_ATL_fsaverage-rh.label'),
                          subject=subject)
    rATL.name='rATL'
    PVA = mne.read_label(path.join(labels_path, 'PVA_fsaverage-lh.label'),
                          subject=subject)
    PVA.name='PVA'
    IFG = mne.read_label(path.join(labels_path, 'IFG_fsaverage-lh.label'),
                          subject=subject)
    IFG.name='IFG'
    AG = mne.read_label(path.join(labels_path, 'AG_fsaverage-lh.label'),
                          subject=subject)
    AG.name='AG'
    PTC = mne.read_label(path.join(labels_path, 'PTC_fsaverage-lh.label'),
                          subject=subject)
    PTC.name='PTC'
    
    rois = [lATL,
            rATL, 
            PVA,
            IFG,
            AG,
            PTC]
    
    n_comp = 5
    stcs_psf_mne, pca_vars_mne = get_point_spread(
        rm_mne, src, rois, mode='pca', n_comp=n_comp, norm=None,
        return_pca_vars=True)
    
    n_verts = rm_mne.shape[1]
    
    
    label_names = [
            lATL.name,
            rATL.name, 
            PVA.name,
            IFG.name,
            AG.name,
            PTC.name]
    
    with np.printoptions(precision=1):
        for [name, var] in zip(label_names, pca_vars_mne):
            print(f'{name}: {var.sum():.1f}% {var}')
    
    n_labels = len(label_names)
    
    psfs_mat = np.zeros([n_labels, n_verts])
    for [i, s] in enumerate(stcs_psf_mne):
        psfs_mat[i, :] = s.data[:, 0]
    # Compute label-to-label leakage as Pearson correlation of PSFs
    # Sign of correlation is arbitrary, so take absolute values
    leakage_mne = np.abs(np.corrcoef(psfs_mat))
    overall.append(leakage_mne)
   
overall = np.stack(overall)
overall.shape
avg_leak = overall.mean(axis=0)
import seaborn as sns
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
    
node_order = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
node_angles = circular_layout(label_names, node_order, start_pos=90,
                              group_boundaries=[0, len(label_names) / 2])

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
fig = plot_connectivity_circle(avg_leak, label_names, n_lines=300,
                         node_angles=node_angles, node_colors=label_colors,
                         title='Leakage ROIs', ax=ax)

fig, ax = plt.subplots(figsize=(8, 8))
fig = sns.heatmap(avg_leak,annot=True, xticklabels=label_names,
            yticklabels=label_names, cmap='viridis')





