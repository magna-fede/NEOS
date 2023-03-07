#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:25:59 2023

@author: fm02
"""

import NEOS_config as config
import sys

from os import path
import numpy as np
import pandas as pd

import mne
#from mne.viz import plot_compare_evokeds
from mne.stats import linear_regression

# import seaborn as sns
   
def get_participant_betas(sbj_id):       
    # sentences = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/Sentences_forMEG_factorised.txt', sep='\t', header=0)
    # stimuli_ALL = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/stimuli_all_onewordsemsim.csv', sep=',')
    # stimuli_ALL = stimuli_ALL.drop(['ID', 'Sentence'], axis=1)
    # sentences = sentences.drop(['Question', 'Ans'], axis = 1)
    # new = pd.merge(sentences, stimuli_ALL, on='Word')
    meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv', header=0)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels[sbj_id]['eeg']
    
    ovr = config.ovr_procedure
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels[sbj_id]['eeg']
    
    if ovr[sbj_id] == 'ovrons':
        over = '_ovrwonset'
    elif ovr[sbj_id] == 'ovr':
        over = '_ovrw'
    elif ovr[sbj_id] == 'novr':
        over = ''
    condition = 'both'
    
    raw_test = []   
    
    for i in range(1,6):
        raw_test.append(mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_ica{over}_{condition}_raw.fif")))
    
    raw_test= mne.concatenate_raws(raw_test)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
    
    raw_test.interpolate_bads(reset_bads=True)
    raw_test.filter(l_freq=0.5, h_freq=None)
    
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
    
    rows = np.where(target_evts[:,2]==999)[0]
    
    event_dict = {'FRP': 999}
    
    tmin, tmax = -.3, .7
        
        # regular epoching
    
    epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
                        reject=None, preload=True)
    
    metadata = pd.DataFrame(columns=meta.columns)
    
    for row in rows: 
        index = target_evts[row-2, 2]*100 + target_evts[row-1, 2]
        metadata = pd.concat([metadata,
                              meta[meta['ID']==index]])
    
    epochs.metadata = metadata
    
    factors = ['ConcM', 'Sim']
    
    for factor in factors:
        df = epochs.metadata.copy()
        # df[factor] = pd.cut(df[factor], 4, labels=False)
        
        # colors = {str(val): val for val in df[factor].unique()}    
        epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
        # evokeds = {val: epochs[factor + " == " + val].average() for val in colors}
        # plot_compare_evokeds(evokeds, colors=colors, split_legend=True,
        #                      cmap=(factor + " Percentile", "viridis"))	               
        names = ["Intercept", factor]
        res = linear_regression(epochs, epochs.metadata[names], names=names)
        # for cond in names:
            # res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit='s'),
            #                           topomap_args=dict(time_unit='s'))
        mne.write_evokeds(f"/imaging/hauk/users/fm02/MEG_NEOS/data/AVE/{sbj_id}_evokedbetas_{factor}.fif",
                          res[factor].beta, overwrite=True)  
    
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 30) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    get_participant_betas(ss)   














