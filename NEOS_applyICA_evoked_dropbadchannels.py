#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:15 2023

@author: fm02
"""

import matplotlib.pyplot as plt

import mne
from mne.stats.regression import linear_regression_raw


import sys
import os
from os import path

import numpy as np
import pandas as pd

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from my_eyeCA import apply_ica

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

def ovr_sub(ovr):
    if ovr in ['nover', 'novr', 'novrw']:
        ovr = ''
    elif ovr in ['ovrw', 'ovr', 'over', 'overw']:
        ovr = '_ovrw'
    elif ovr in ['ovrwonset', 'ovrons', 'overonset']:
        ovr = '_ovrwonset'
    return ovr

# %%
# Set MNE's log level to DEBUG
def create_evoked_from_raw(sbj_id):
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw_test = apply_ica.get_ica_raw(sbj_id, 
                                     condition='both',
                                     overweighting=ovr,
                                     interpolate=False, 
                                     drop_EEG_4_8=False)
    
    raw_test = raw_test.set_eeg_reference(ref_channels='average', projection=True)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
       
    picks = mne.pick_types(raw_test.info, meg=True, eeg=True, exclude='bads')
    
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
            
    rows = np.where(target_evts[:,2]==999)[0]
    for row in rows:
        if target_evts[row-2, 2] == 1:
            target_evts[row, 2] = 991
        elif target_evts[row-2, 2] == 2:
            target_evts[row, 2] = 992
        elif target_evts[row-2, 2] == 3:
            target_evts[row, 2] = 993
        elif target_evts[row-2, 2] == 4:
            target_evts[row, 2] = 994
        elif target_evts[row-2, 2] == 5:
            target_evts[row, 2] = 995
            
    event_dict = {'Abstract/Predictable': 991, 
                  'Concrete/Predictable': 992,
                  'Abstract/Unpredictable': 993, 
                  'Concrete/Unpredictable': 994}
    tmin, tmax = -.3, .7
    
    # regular epoching

    epochs = mne.Epochs(raw_test, target_evts, event_dict, picks=picks,
                        tmin=tmin, tmax=tmax, reject=reject_criteria, 
                        preload=True)

    # evokeds = linear_regression_raw(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
    #                     reject=reject_criteria)

    evokeds = list()    

    for key in event_dict.keys():
        evokeds.append(epochs[key].average())
    
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evokeds_dropbads.fif", evokeds, overwrite=True)

def create_evoked_from_ICA_raw(sbj_id):
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)    
    
    raw.info['bads'] = bad_eeg
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
    
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
            
    rows = np.where(target_evts[:,2]==999)[0]
    for row in rows:
        if target_evts[row-2, 2] == 1:
            target_evts[row, 2] = 991
        elif target_evts[row-2, 2] == 2:
            target_evts[row, 2] = 992
        elif target_evts[row-2, 2] == 3:
            target_evts[row, 2] = 993
        elif target_evts[row-2, 2] == 4:
            target_evts[row, 2] = 994
        elif target_evts[row-2, 2] == 5:
            target_evts[row, 2] = 995
            
    event_dict = {'Abstract/Predictable': 991, 
                  'Concrete/Predictable': 992,
                  'Abstract/Unpredictable': 993, 
                  'Concrete/Unpredictable': 994}
    tmin, tmax = -.2, .5

    # regular epoching

    epochs = mne.Epochs(raw, target_evts, event_dict, picks=picks,
                        tmin=tmin, tmax=tmax, reject=reject_criteria, 
                        preload=True)

    # evokeds = linear_regression_raw(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
    #                     reject=reject_criteria)
    



    for condition in ['Predictable', 'Unpredictable', 'Concrete', 'Abstract']:
        evoked = epochs[condition].average()
    
        mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_{condition}_evokeds_dropbads-ave.fif", evoked, overwrite=True)

def create_evoked_from_ICA_raw_keepallchannels(sbj_id):
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)    
    
    #raw.info['bads'] = bad_eeg
    picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude=[])
    
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
            
    rows = np.where(target_evts[:,2]==999)[0]
    for row in rows:
        if target_evts[row-2, 2] == 1:
            target_evts[row, 2] = 991
        elif target_evts[row-2, 2] == 2:
            target_evts[row, 2] = 992
        elif target_evts[row-2, 2] == 3:
            target_evts[row, 2] = 993
        elif target_evts[row-2, 2] == 4:
            target_evts[row, 2] = 994
        elif target_evts[row-2, 2] == 5:
            target_evts[row, 2] = 995
            
    event_dict = {'Abstract/Predictable': 991, 
                  'Concrete/Predictable': 992,
                  'Abstract/Unpredictable': 993, 
                  'Concrete/Unpredictable': 994}
    tmin, tmax = -.2, .5

    # regular epoching

    epochs = mne.Epochs(raw, target_evts, event_dict, picks=picks,
                        tmin=tmin, tmax=tmax, reject=reject_criteria, 
                        preload=True)

    # evokeds = linear_regression_raw(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
    #                     reject=reject_criteria)
    



    for condition in ['Predictable', 'Unpredictable', 'Concrete', 'Abstract']:
        evoked = epochs[condition].average()
    
        mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_{condition}_evokeds_-ave.fif", evoked, overwrite=True)
    
    
        
    
    