#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:51:16 2023

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

import re


ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")
snr = 3.
lambda2 = 1. / snr ** 2
# orientation = 'normal'
# conditions = [['Predictable', 'Unpredictable'], ['Abstract', 'Concrete']]
# conditions = ['abspred', 'absunpred', 'concpred', 'concunpred']

conditions = ['Predictable', 'Unpredictable', 'Abstract', 'Concrete']

def compute_stcs(sbj_id, method="MNE", inv_suf='', orientation=None):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    evs=dict()
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evokeds_nochan_4_8.fif")
        evokeds = mne.read_evokeds(fname)
        ev_0 = list()
        ev_1 = list()
        for e, evoked in enumerate(evokeds):
            if re.match(f'([a-z]*\/)?{condition[0]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_0.append(evoked)
            elif re.match(f'([a-z]*\/)?{condition[1]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_1.append(evoked)  
                
        evs[condition[0]] = mne.combine_evoked(ev_0, weights='nave')
        evs[condition[1]] = mne.combine_evoked(ev_1, weights='nave')
        
    for ev in evs.keys():
        stc = mne.minimum_norm.apply_inverse(evs[ev], inverse_operator,
                                             lambda2, method=method,
                                             pick_ori=orientation, verbose=True)
        if len(inv_suf)==0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}")
        elif len(inv_suf)>0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}_{inv_suf}")
        stc.save(stc_fname)
        
def compute_stcs_dropbads(sbj_id, method="MNE", inv_suf='', orientation=None):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    evs=dict()
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evokeds_dropbads.fif")
        evokeds = mne.read_evokeds(fname)
        ev_0 = list()
        ev_1 = list()
        for e, evoked in enumerate(evokeds):
            if re.match(f'([a-z]*\/)?{condition[0]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_0.append(evoked)
            elif re.match(f'([a-z]*\/)?{condition[1]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_1.append(evoked)  
                
        evs[condition[0]] = mne.combine_evoked(ev_0, weights='nave')
        evs[condition[1]] = mne.combine_evoked(ev_1, weights='nave')
        
    for ev in evs.keys():
        stc = mne.minimum_norm.apply_inverse(evs[ev], inverse_operator,
                                             lambda2, method=method,
                                             pick_ori=orientation, verbose=True)
        if len(inv_suf)==0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}")
        elif len(inv_suf)>0:
            stc_fname = path.join(stc_path, f"{subject}_stc_{ev}_{method}_{inv_suf}")
        stc.save(stc_fname)

def stcs_inlabel_dropbads(sbj_id, method="MNE", inv_suf='', orientation=None, mode_avg='mean_flip'):
    
    labels_path = path.join(config.data_path, "my_ROIs")

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

    rois_lab = ['lATL',
                'rATL', 
                'PVA',
                'IFG',
                'AG',
                'PTC']
            
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    evs=dict()
    for condition in conditions:
        fname = path.join(ave_path, f"{subject}_evokeds_dropbads.fif")
        evokeds = mne.read_evokeds(fname)
        [evoked.resample(250, npad='auto') for evoked in evokeds]
        ev_0 = list()
        ev_1 = list()
        for e, evoked in enumerate(evokeds):
            if re.match(f'([a-z]*\/)?{condition[0]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_0.append(evoked)
            elif re.match(f'([a-z]*\/)?{condition[1]}(\/[a-z]*)?', evoked.comment, re.IGNORECASE):
                ev_1.append(evoked)  
                
        evs[condition[0]] = mne.combine_evoked(ev_0, weights='nave')
        evs[condition[1]] = mne.combine_evoked(ev_1, weights='nave')
    
    rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                    subject_from='fsaverage', 
                                    subjects_dir=config.subjects_dir)

    for ev in evs.keys():
        stc = mne.minimum_norm.apply_inverse(evs[ev], inverse_operator,
                                             lambda2, method=method,
                                             pick_ori=orientation, verbose=True)
        
        src = inverse_operator["src"]
        
        activity = stc.extract_label_time_course(rois_subject, src, mode=mode_avg)
        activity = pd.DataFrame(activity.T, columns=rois_lab)
        activity.to_csv(path.join(ave_path, "in_labels", f"{sbj_id}_{ev}_in_labels.csv"))
        
        
def save_stcs_condition(fpath_evoked, inverse_operator, method, 
                        inv_suf, orientation=None):
    
    evoked = mne.read_evokeds(fpath_evoked)[0]
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator,
                                         lambda2, method=method,
                                         pick_ori=orientation, verbose=True)
    # THIS WORKS ONLY IF YOU USE THE STANDARD NAMING, USE WITH CARE
    # ideally you'd want to use evoked.comment
    
    subject, condition = fpath_evoked.split('/')[-1].split('_')[0:2]
    
    stc_fname = path.join(stc_path, f"{subject}_stc_{condition}_{method}_{inv_suf}")
    stc.save(stc_fname)    
    return stc

def save_unfold_stcs_condition(fpath_evoked, inverse_operator, method, 
                        inv_suf, orientation=None):
    
    evoked = mne.read_evokeds(fpath_evoked)[0]
    #evoked.apply_baseline(baseline=(-0.152,-0.020))
    
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator,
                                         lambda2, method=method,
                                         pick_ori=orientation, verbose=True)
    # THIS WORKS ONLY IF YOU USE THE STANDARD NAMING, USE WITH CARE
    # ideally you'd want to use evoked.comment
    
    subject, condition = fpath_evoked.split('/')[-1].split('_')[0:2]
    
    stc_fname = path.join(stc_path, f"{subject}_unfold_stc_{condition}_{method}_{inv_suf}")
    stc.save(stc_fname, overwrite=True)    
    return stc

def compute_evoked_condition_stcs(sbj_id, method="eLORETA", inv_suf='empirical_dropbads',
                          orientation=None):
    
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    # PARTICIPANT 12 PROBLEMS WITH EEG DURING RECORDING
    if sbj_id==12:
        inv_fname = path.join(sbj_path, subject + f'_MEG{inv_suf}-inv.fif')
    else:
        inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)    
    for condition in ['Abstract', 'Concrete', 'Predictable', 'Unpredictable']:
        filename = path.join(ave_path, f"{subject}_{condition}_evokeds_dropbads-ave.fif")
        save_stcs_condition(filename, inverse_operator, method, inv_suf, orientation)
 
def compute_unfold_evoked_condition_stcs(sbj_id, method="eLORETA", inv_suf='empirical_dropbads',
                          orientation=None):
    
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    # PARTICIPANT 12 PROBLEMS WITH EEG DURING RECORDING
    if sbj_id==12:
        inv_fname = path.join(sbj_path, subject + f'_MEG{inv_suf}-inv.fif')
    else:
        inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)    
    for condition in ['Abstract', 'Concrete', 'Predictable', 'Unpredictable']:
        filename = path.join(ave_path, f"{subject}_{condition}_unfold_evoked-ave.fif")
        save_unfold_stcs_condition(filename, inverse_operator, method, inv_suf, orientation)

       

def stcs_inlabel_from_stc(sbj_id, method="eLORETA", inv_suf='empirical_dropbads', mode_avg='mean'):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    labels_path = path.join(config.data_path, "my_ROIs")
    
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
    
    rois_lab = ['lATL',
                'rATL', 
                'PVA',
                'IFG',
                'AG',
                'PTC']
    
    if sbj_id==12:
        inv_fname = path.join(sbj_path, subject + f'_MEG{inv_suf}-inv.fif')
    else:
        inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    src = inverse_operator["src"]
    
    subject = str(sbj_id)
    rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                    subject_from='fsaverage', 
                                    subjects_dir=config.subjects_dir)
    stc = dict.fromkeys(conditions)
    activity = dict.fromkeys(conditions)
    
    for condition in conditions:
        stc[condition] = mne.read_source_estimate(os.path.join(stc_path, f"{subject}_stc_{condition}_{method}_{inv_suf}"))
        stc[condition].resample(250)
    
        activity[condition] = stc[condition].extract_label_time_course(rois_subject, src=src, mode=mode_avg)

        activity[condition] = pd.DataFrame(activity[condition].T, columns=rois_lab, index=stc[condition].times)
        activity[condition].to_csv(path.join(stc_path, "in_labels", f"{sbj_id}_{condition}_in_labels.csv"))

def stcs_unfold_inlabel_from_stc(sbj_id, method="eLORETA", inv_suf='empirical_dropbads', mode_avg='mean'):
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    labels_path = path.join(config.data_path, "my_ROIs")
    
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
    
    rois_lab = ['lATL',
                'rATL', 
                'PVA',
                'IFG',
                'AG',
                'PTC']
    
    if sbj_id==12:
        inv_fname = path.join(sbj_path, subject + f'_MEG{inv_suf}-inv.fif')
    else:
        inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
        
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    src = inverse_operator["src"]
    
    subject = str(sbj_id)
    rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                    subject_from='fsaverage', 
                                    subjects_dir=config.subjects_dir)
    stc = dict.fromkeys(conditions)
    activity = dict.fromkeys(conditions)
    
    for condition in conditions:
        stc[condition] = mne.read_source_estimate(os.path.join(stc_path, f"{subject}_unfold_stc_{condition}_{method}_{inv_suf}"))

        activity[condition] = stc[condition].extract_label_time_course(rois_subject, src=src, mode=mode_avg)

        activity[condition] = pd.DataFrame(activity[condition].T, columns=rois_lab, index=stc[condition].times)
        activity[condition].to_csv(path.join(stc_path, "in_labels", f"{sbj_id}_unfold_{condition}_in_labels.csv"))
        
# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]


# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     compute_stcs(ss)    
    