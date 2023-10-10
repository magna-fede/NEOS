#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:51:19 2023

@author: fm02
"""

import sys
import os
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)
import pickle

import mne
from mne.minimum_norm import apply_inverse_epochs

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

#os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import apply_ica

os.chdir("/home/fm02/MEG_NEOS/NEOS")

mne.viz.set_browser_backend("matplotlib")

snr = 1.0  # snr should be 1 for single epoch inversion
lambda2 = 1.0 / snr ** 2
loose = 0.2
depth = None

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

def predictability(x):
    if (x==991) or (x==992):
        return "Predictable"
    elif (x==993) or (x==994):
        return "Unpredictable"
    else:
        return "Error"
    
def concreteness(x):
    if (x==991) or (x==993):
        return "Abstract"
    elif (x==992) or (x==994):
        return "Concrete"
    else:
        return "Error"


# %%

def get_decoding_sensor_scores(sbj_id):
    
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)  

    raw.info['bads'] = bad_eeg
    
    if sbj_id==12:
        picks = mne.pick_types(raw.info, meg=True, eeg=False)
    else:
        picks = mne.pick_types(raw.info, meg=True, eeg=True)
        
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
   
    epochs = mne.Epochs(raw, target_evts, event_dict, tmin=tmin, tmax=tmax,
                       picks=picks, reject=reject_criteria, preload=True)
   
    epochs.resample(250, npad='auto')
    X = epochs.get_data() 

    evs = epochs.events[:, 2]
    y_conc = pd.Series(evs).apply(concreteness)

    y_pred = pd.Series(evs).apply(predictability)
    
     # prepare a series of classifier applied at each time sample
    clf = make_pipeline(
        Scaler(epochs.info),
        Vectorizer(),
        LinearModel(LogisticRegression(C=1, solver='liblinear',
                                       max_iter=1000)),
    )
    time_decod = SlidingEstimator(clf, scoring="roc_auc")
    
    # Run cross-validated decoding analyses:

    scores = dict.fromkeys(['Concreteness', 'Predictability'])
    for y, cond in zip([y_conc, y_pred],
                       ['Concreteness', 'Predictability']):
        scores[cond] = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
        
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/sensor_space/{sbj_id}_scores_sensor.P', 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_decoding_sensor_avg3trials_scores(sbj_id):
    
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)  

    raw.info['bads'] = bad_eeg
    
    if sbj_id==12:
        picks = mne.pick_types(raw.info, meg=True, eeg=False)
    else:
        picks = mne.pick_types(raw.info, meg=True, eeg=True)
        
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

    if sbj_id==12:
        epochs = mne.Epochs(raw, target_evts, event_dict, tmin=tmin, tmax=tmax,
                           picks=picks, 
                           reject={x: reject_criteria[x] for x in ['grad', 'mag'] if x in reject_criteria},
                           preload=True)
    else:
        epochs = mne.Epochs(raw, target_evts, event_dict, tmin=tmin, tmax=tmax,
                       picks=picks, reject=reject_criteria, preload=True)
   
    epochs.resample(250, npad='auto')
    X = epochs.get_data() 

    evs = epochs.events[:, 2]
    y_conc = pd.Series(evs).apply(concreteness)

    y_pred = pd.Series(evs).apply(predictability)
    
    X_a = X[y_conc=='Abstract',:,:]
    X_c = X[y_conc=='Concrete',:,:]
    
    X_p = X[y_pred=='Predictable',:,:]
    X_u = X[y_pred=='Unpredictable',:,:]
    
    trials = dict()
    trials['Abstract'] = X_a
    trials['Concrete'] = X_c
    trials['Predictable'] = X_p
    trials['Unpredictable'] = X_u

    trials_avg3 = dict.fromkeys(trials.keys())
    
    for task in trials.keys():


        while len(trials[task])%3 != 0:
            trials[task] = np.delete(trials[task], 
                                    len(trials[task])-1, 0)
            # split data in groups of 3 trials
        new_tsk = np.vsplit(trials[task], len(trials[task])/3)
        new_trials = []
        # calculate average for each timepoint (axis=0) of the 3 trials
        for nt in new_tsk:
            new_trials.append(np.mean(np.array(nt),0))
        # assign group to the corresponding task in the dict
        # each is 3D array n_trial*n_vertices*n_timepoints
        
        trials_avg3[task] = np.array(new_trials)
    
    X_conc = np.concatenate([trials_avg3['Abstract'],
                        trials_avg3['Concrete']])
    y_conc = np.array(['Abstract']*len(trials_avg3['Abstract']) + \
                      ['Concrete']*len(trials_avg3['Concrete']))

    X_pred = np.concatenate([trials_avg3['Unpredictable'],
                        trials_avg3['Predictable']])
    y_pred = np.array(['Unpredictable']*len(trials_avg3['Unpredictable']) + \
                      ['Predictable']*len(trials_avg3['Predictable']))

        
     # prepare a series of classifier applied at each time sample
    clf = make_pipeline(
        Scaler(epochs.info),
        Vectorizer(),
        LinearModel(LogisticRegression(C=1, solver='liblinear',
                                       max_iter=1000)),
    )
    time_decod = SlidingEstimator(clf, scoring="roc_auc")
    
    # Run cross-validated decoding analyses:

    scores = dict.fromkeys(['Concreteness', 'Predictability'])
    for X, y, cond in zip([X_conc, X_pred],
                          [y_conc, y_pred],
                          ['Concreteness', 'Predictability']):
        X, y = shuffle(X, y,
               # random_state=0
               )    
        scores[cond] = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=-1)
        
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/sensor_space/{sbj_id}_scores_3pseudotrials_sensor.P', 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)            