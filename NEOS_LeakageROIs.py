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

import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

all_leak = list()
all_leak_norm = list()

sbj_ids = [1,2]

for sbj_id in sbj_ids:
    subject = str(sbj_id)
    subjects_dir = config.subjects_dir
    
    #### THIS MUST BE CONVERTED TO DATA_PATH NOT DATA***OLD***PATH
    sbj_path = path.join(config.dataold_path, config.map_subjects[sbj_id][0])
    
    info = mne.io.read_info(path.join(sbj_path, "block1_sss_f_raw.fif"))  
    
    fwd_fname = path.join(sbj_path, subject + '_EEGMEG-fwd.fif')    
    cov_fname =   fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] \
                                        + '_covariancematrix_empirical-cov.fif')
    inv_fname = path.join(sbj_path, subject + '_EEGMEG-inv_emp3150.fif')
    
    cov = mne.read_cov(cov_fname)
    
    forward_loose = mne.read_forward_solution(fwd_fname)
    forward_fixed = forward_loose.copy()
    forward_fixed = mne.convert_forward_solution(
        forward_fixed, surf_ori=True, force_fixed=True, copy=False)
    
    inverse_operator_loose = mne.minimum_norm.read_inverse_operator(inv_fname)
    inverse_operator_fixed = mne.minimum_norm.make_inverse_operator(info, forward_fixed, noise_cov=cov,
                                   fixed=True, loose=0., depth=None,
                                   verbose=None)
    
    rm_mne = make_inverse_resolution_matrix(forward_loose, inverse_operator_loose,
                                            method='eLORETA', lambda2=1. / 3.**2)
    
    src = inverse_operator_fixed['src']
    
    #### THIS MUST BE CONVERTED TO DATA_PATH NOT DATA***OLD***PATH
    labels_path = path.join(config.dataold_path, "my_ROIs")
    
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
    
    rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                    subject_from='fsaverage', 
                                    subjects_dir=config.subjects_dir)

    n_verts = forward_loose['nsource']    
    label_names = [
            lATL.name,
            rATL.name, 
            PVA.name,
            IFG.name,
            AG.name,
            PTC.name]
    n_labels = len(label_names)

# %% THIS IS FOR PLOTTING FIRST 5 PC on BRAIN
    
    n_comp = 5
    stcs_psf_mne, pca_vars_mne = get_point_spread(
        rm_mne, src, rois_subject, mode='pca', n_comp=n_comp, norm=None,
        return_pca_vars=True)

    
    with np.printoptions(precision=1):
        for [name, var] in zip(label_names, pca_vars_mne):
            print(f'{name}: {var.sum():.1f}% {var}')
    
    Brain = mne.viz.get_brain_class()
    
    brain = Brain('2', 'both', 'inflated', subjects_dir=subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))
    [brain.add_label(roi) for roi in rois_subject]
    for i in range(0, n_labels):
        brain_psf = stcs_psf_mne[i].plot(subject, 'inflated', 'lh', subjects_dir=subjects_dir)

 # %% THIS IS FOR ACTUAL LEAKAGE COMPUTATION
    stcs_psf_mne = get_point_spread(
        rm_mne, src, rois_subject, mode=None, norm=None)
     
    psfs_mat = np.zeros([n_labels, n_verts])
    for [i, s] in enumerate(stcs_psf_mne):
        psfs_mat[i, :] = s.data[:, 0]
    # Compute label-to-label leakage as Pearson correlation of PSFs
    # Sign of correlation is arbitrary, so take absolute values
    # leakage_mne = np.abs(np.corrcoef(psfs_mat))
    # overall.append(leakage_mne)
    
    
    stcs_psf = get_point_spread(
    rm_mne, src, rois_subject, mode=None, norm='norm',
    return_pca_vars=False)
    
    leakage = np.zeros([6, 6])
    leakage_norm = np.zeros([6, 6])

    for r in np.arange(0, len(rois_subject)):
        # stc = stcs_psf[r]
        # brain = stc.plot(
        #     subjects_dir=subjects_dir, subject='fsaverage', hemi='lh', views='lateral')
        # brain.add_text(0.1, 0.9, label_names[r], 'title', font_size=16)
        # brain.add_label(SN_ROI[r], borders=True,color='g')
    
        # stc_all[r]=stc_all[r]+ stc
        for [c, label] in enumerate(rois_subject):
        # for [c, label] in enumerate(labels):
    
            stc_label = stcs_psf[r].in_label(rois_subject[c])
            leakage[r, c] = np.mean(np.abs(stc_label.data))
            
    for r in np.arange(0, len(rois_subject)):
        leakage_norm[:, r] = leakage[:, r].copy()/leakage[r, r]
    
    all_leak.append(leakage)
    all_leak_norm.append(leakage_norm)
   
avg_leak_norm = np.stack(all_leak_norm).mean(axis=0)
sns.heatmap(avg_leak_norm, annot=True, cmap='viridis', xticklabels=label_names, yticklabels=label_names)
   
# overall = np.stack(overall)
# overall.shape
# avg_leak = overall.mean(axis=0)
# import seaborn as sns
# label_colors = sns.color_palette(['#FFBE0B',
#                             '#FB5607',
#                             '#FF006E',
#                             '#8338EC',
#                             '#3A86FF',
#                             '#1D437F',
#                             '#1D437F'
#                             ])
# label_ypos = list()
# for name in label_names:
#     idx = label_names.index(name)
#     ypos = np.mean(rois[idx].pos[:, 1])
#     label_ypos.append(ypos)
    
# node_order = ['lATL', 'rATL', 'PTC', 'IFG', 'AG', 'PVA']
# node_angles = circular_layout(label_names, node_order, start_pos=90,
#                               group_boundaries=[0, len(label_names) / 2])

# fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
#                        subplot_kw=dict(polar=True))
# fig = plot_connectivity_circle(avg_leak, label_names, n_lines=300,
#                          node_angles=node_angles, node_colors=label_colors,
#                          title='Leakage ROIs', ax=ax)

# fig, ax = plt.subplots(figsize=(8, 8))
# fig = sns.heatmap(avg_leak,annot=True, xticklabels=label_names,
#             yticklabels=label_names, cmap='viridis')





