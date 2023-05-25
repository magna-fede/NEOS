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
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)

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
   
    tmin, tmax = -.3, .7
       
    epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
                       picks=picks, reject=None, preload=True)
   
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

    scores = dict.fromkeys(['Concreteness', 'Predidctability'])
    for y, cond in zip([y_conc, y_pred],
                       ['Concreteness', 'Predictability']):
        scores[cond] = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=None)
        
    with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/source_space/{sbj_id}_scores_sensor.P', 'wb') as handle:
        pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

            