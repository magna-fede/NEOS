#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:36:53 2023

@author: fm02
"""

import sys
import os
from os import path

import numpy as np
import pandas as pd


import pickle

import mne
from mne.minimum_norm import apply_inverse_epochs

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

#os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import apply_ica

import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/home/fm02/MEG_NEOS/NEOS")
snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
loose = 0.2
depth = None
reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

labels_path = path.join(config.data_path, "my_ROIs")
stc_path = path.join(config.data_path, "stcs")

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

method='eLORETA'
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

stc= dict()
stc_conditions = ['Intercept',
            'ConcM', 
            'Sim',
            'ConcPred']

for cond in stc_conditions:
    stc[cond] = list()
    
activity=dict()
rois_lab = ['lATL',
            'rATL', 
            'PVA',
            'IFG',
            'AG',
            'PTC']

for roi in rois_lab:
    activity[roi] = dict()
    for cond in stc_conditions:
        activity[roi][cond] = list()

# thisi s wrong because nave=1 so scaling is off
for sbj_id in sbj_ids:
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + '_EEGMEG-inv_emp3150.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)

    for cond in stc_conditions:
        evo_fname = f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked-tvals_forsource.fif"
        evoked = mne.read_evokeds(evo_fname, condition=cond, verbose="ERROR")
        stc_c = mne.minimum_norm.apply_inverse(evoked, inverse_operator,
                                         lambda2, method=method,
                                         pick_ori=None, verbose=True)
        stc[cond].append(stc_c)
        rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                        subject_from='fsaverage', 
                                        subjects_dir=config.subjects_dir)
    
        for i,roi in enumerate(rois_lab):
            activity[roi][cond].append(stc_c.extract_label_time_course(rois_subject[i], inverse_operator['src'],
                                                     mode='mean'))
            
for cond in stc_conditions:
    fig=plt.subplots(1)
    for roi in rois_lab:   
        sns.lineplot(x=np.linspace(-300,700,250), y=np.concatenate(activity[roi][cond]).mean(axis=0), label=roi)
    plt.title(cond)
    plt.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/data/plots/tvals_insource/{cond}_acrossROIs.png")
    
for cond in stc_conditions:
    for roi in rois_lab:
        fig=plt.subplots(1)
        for i in activity[roi][cond]:
            sns.lineplot(x=np.linspace(-300,700,250), y=i.squeeze(), color='k', linewidth=0.1, alpha=0.5)
        plt.title(roi)
        plt.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/data/plots/tvals_insource/{cond}_in_{roi}_acrossparticipants.png")
        