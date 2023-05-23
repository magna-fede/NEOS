#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:40:24 2023

@author: fm02
"""
import sys
import os
from os import path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

import pickle

import mne
from mne.minimum_norm import apply_inverse_epochs

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from mne.decoding import cross_val_multiscore, LinearModel, SlidingEstimator

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

meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv', header=0)
pred = ['ID', 'ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']
meta = meta[pred]

est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
est.fit(meta[['ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']])
Xt = est.transform(meta[['ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']])

cols = ['ConcCont', 'Length', 'Zipf', 'Position', 'PredCont']

# %%

def decoding_continuous_predictors(sbj_id, method='eLORETA', inv_suf='shrunk_dropbads'):
    
    subject = str(sbj_id)
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
     
    event_dict = {'FRP': 999}
     
    tmin, tmax = -.3, .7
       
    epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
                       picks=picks, reject=None, preload=True)
     
    metadata = pd.DataFrame(columns=meta.columns)
     
    for row in rows: 
        index = target_evts[row-2, 2]*100 + target_evts[row-1, 2]
        metadata = pd.concat([metadata,
                             meta[meta['ID']==index]])
     
    epochs.metadata = metadata
    
    y = Xt[metadata.index,:]
    
    epochs.resample(250, npad='auto')
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
    
    rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                    subject_from='fsaverage', 
                                    subjects_dir=config.subjects_dir)
  
    rois_lab = ['lATL',
                  'rATL', 
                  'PVA',
                  'IFG',
                  'AG',
                  'PTC']
        
    stc = dict()
    X = dict()
    for roi, key in zip(rois_subject, rois_lab):
        stc[key] = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, label=roi,
                                pick_ori="normal", nave=len(epochs))
        X[key] = np.array([s.data for s in stc[key]])


     # prepare a series of classifier applied at each time sample
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        LinearModel(LogisticRegression(C=1, solver='lbfgs',
                                       max_iter=1000)),
    )
    time_decod = SlidingEstimator(clf, scoring='roc_auc_ovr')
    
    # Run cross-validated decoding analyses:

    scores = dict.fromkeys(rois_lab)
    for i, cond in enumerate(cols):
        for roi in rois_lab:
            y_cond = y[:,i]
            scores[roi] = cross_val_multiscore(time_decod, X[roi], y_cond, cv=5, n_jobs=None)
        
        with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/source_space/{sbj_id}_scores_{cond}.P', 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_decoding_scores(sbj_id, method='eLORETA', inv_suf='shrunk_dropbads'):
    
    subject = str(sbj_id)
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
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
    inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)

    rois_subject = mne.morph_labels(rois, subject_to=subject, 
                                    subject_from='fsaverage', 
                                    subjects_dir=config.subjects_dir)
    
    rois_lab = ['lATL',
                'rATL', 
                'PVA',
                'IFG',
                'AG',
                'PTC']
    
    stc = dict()
    X = dict()
    for roi, key in zip(rois_subject, rois_lab):
        stc[key] = apply_inverse_epochs(epochs, inverse_operator, lambda2, method, label=roi,
                                pick_ori="normal", nave=len(epochs))
        X[key] = np.array([s.data for s in stc[key]])

    evs = epochs.events[:, 2]
    y_conc = pd.Series(evs).apply(concreteness)

    y_pred = pd.Series(evs).apply(predictability)
    
     # prepare a series of classifier applied at each time sample
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        LinearModel(LogisticRegression(C=1, solver='liblinear',
                                       max_iter=1000)),
    )
    time_decod = SlidingEstimator(clf, scoring="roc_auc")
    
    # Run cross-validated decoding analyses:

    scores = dict.fromkeys(rois_lab)
    for y, cond in zip([y_conc, y_pred],
                       ['Concreteness', 'Predictability']):
        for roi in rois_lab:
            scores[roi] = cross_val_multiscore(time_decod, X[roi], y, cv=5, n_jobs=None)
        
        with open(f'/imaging/hauk/users/fm02/MEG_NEOS/data/Decoding/source_space/{sbj_id}_scores_{cond}.P', 'wb') as handle:
            pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]


# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     make_InverseOperator(ss)    
    