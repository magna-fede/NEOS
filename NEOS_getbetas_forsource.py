#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:25:59 2023

@author: fm02
"""

import sys
import os
from os import path
import numpy as np
import pandas as pd

import mne
#from mne.viz import plot_compare_evokeds
from mne.stats import linear_regression

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

from my_eyeCA import apply_ica
from sklearn.preprocessing import StandardScaler
# import seaborn as sns

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

meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv',
                   header=0)

pred = ['ID', 'Word', 'ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']
meta = meta[pred]

scaler = StandardScaler()
meta[['ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']] = scaler.fit_transform(
    meta[['ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']])

# %%
def get_betas(sbj_id, factors=['ConcM', 'Sim']):
    
        
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
    
    ################ try to drop all bad channels ################ 
    # raw_test.interpolate_bads(reset_bads=True)
    # raw_test.drop_channels(['EEG004', 'EEG008'])
    
    # raw_test.drop_channels(bad_eeg)
    # ################  ################  ################  ################ 
  
    # sentences = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/Sentences_forMEG_factorised.txt', sep='\t', header=0)
    # stimuli_ALL = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/stimuli_all_onewordsemsim.csv', sep=',')
    # stimuli_ALL = stimuli_ALL.drop(['ID', 'Sentence'], axis=1)
    # sentences = sentences.drop(['Question', 'Ans'], axis = 1)
    # new = pd.merge(sentences, stimuli_ALL, on='Word')
    
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
    
    rows = np.where(target_evts[:,2]==999)[0]
    
    event_dict = {'FRP': 999}
    
    tmin, tmax = -.3, .7
        
        # regular epoching
    
    epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
                        picks=picks, reject=reject_criteria, preload=True)
    
    metadata = pd.DataFrame(columns=meta.columns)
    
    for row in rows: 
        index = target_evts[row-2, 2]*100 + target_evts[row-1, 2]
        metadata = pd.concat([metadata,
                              meta[meta['ID']==index]])
    
    drop_log = pd.Series(epochs.drop_log)

    good_ones = drop_log!=('IGNORED',)

    kept_log = drop_log[good_ones].reset_index(drop=True)

    len_kl = kept_log.apply(lambda x: len(x))
    metadata.reset_index(drop=True, inplace=True)
    
    metadata = metadata[len_kl==0]
    
    epochs.metadata = metadata
    
    epochs.resample(250, npad='auto')
    
    factors = ['ConcM', 'Sim', 'ConcPred']
    
    df = epochs.metadata.copy()
    df['ConcPred'] = df['ConcM']*df['Sim']
    # df[factor] = pd.cut(df[factor], 4, labels=False)
    
    # colors = {str(val): val for val in df[factor].unique()}    
    epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
    # evokeds = {val: epochs[factor + " == " + val].average() for val in colors}
    # plot_compare_evokeds(evokeds, colors=colors, split_legend=True,
    #                      cmap=(factor + " Percentile", "viridis"))	               
    names = ["Intercept"] + factors
    res = linear_regression(epochs, epochs.metadata[names], names=names)
    evoked_list_betas = list()
    evoked_list_tvals = list()
    
    for cond in names:
        # res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit='s'),
        #                           topomap_args=dict(time_unit='s'))
        evoked_list_betas.append(res[cond].beta)
        evoked_list_betas[-1].comment = cond
        evoked_list_tvals.append(res[cond].t_val)
        evoked_list_tvals[-1].comment = cond
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked-betas_forsource.fif",
                          evoked_list_betas, overwrite=True)  
    mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evoked-tvals_forsource.fif",
                          evoked_list_tvals, overwrite=True)  
                
# if len(sys.argv) == 1:

#     sbj_ids = np.arange(0, 30) + 1

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     get_betas(ss)   














