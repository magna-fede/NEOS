#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:25:59 2023

@author: fm02
"""

import NEOS_config as config
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

ovr = config.ovr_procedure

meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv', header=0)

pred = ['ID', 'Word', 'ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']
meta = meta[pred]

scaler = StandardScaler()
meta[['ConcM', 'LEN', 'LogFreq(Zipf)', 'Position', 'Sim']] = scaler.fit_transform(meta[['ConcM', 
                                                                                        'LEN', 
                                                                                        'LogFreq(Zipf)', 
                                                                                        'Position', 
                                                                                        'Sim']])

# %%
# Set MNE's log level to DEBUG
def get_betas(sbj_id, factors=['ConcM', 'Sim']):
    
        
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    if ovr[sbj_id] == 'ovrons':
        over = '_ovrwonset'
        ica_dir = path.join(sbj_path, 'ICA_ovr_w_onset')
    elif ovr[sbj_id] == 'ovr':
        over = '_ovrw'
        ica_dir = path.join(sbj_path, 'ICA_ovr_w')
    elif ovr[sbj_id] == 'novr':
        over = ''
        ica_dir = path.join(sbj_path, 'ICA')
    condition = 'both'

    raw_test = []
    
    ica_sub = '_sss_f_raw_ICA_extinfomax_0.99_COV_None'
    ica_sub_file = '_sss_f_raw_ICA_extinfomax_0.99_COV_None-ica_eogvar'
            
    for i in range(1, 6):
        ica_fname = path.join(ica_dir,
                              f'block{i}'+ica_sub,
                              f'block{i}'+ica_sub_file) 
        raw = mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_raw.fif"))
        evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          f'_all_events_block_{i}.fif')
        evt = mne.read_events(evt_file)
        # hey, very important to keep overwrite_saved as false, or this will change
        # the raw files saved for checking which approach is best for each participant
        raw_ica, _, _ = apply_ica.apply_ica_pipeline(raw=raw,                                                                  
                        evt=evt, thresh=1.1, method='both',
						ica_filename=ica_fname, overwrite_saved=False)
        raw_test.append(raw_ica)
    raw_test = mne.concatenate_raws(raw_test)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
    
    ################ try to drop all bad channels ################ 
    raw_test.interpolate_bads(reset_bads=True)
    raw_test.drop_channels(['EEG004', 'EEG008'])
    
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
                        picks=['meg', 'eeg'], reject=reject_criteria, preload=True)
    
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














