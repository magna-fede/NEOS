#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:02:20 2023

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

import h5py

def divide_before_after_target(evts):  
    """I'm not sure this is worth it.
    It would make the modelling of the fixation post target more noisy
    because the target word is always towards the end of the sentence.
    Since we care more about correcting the effects of fixations 
    following that on target, it is probably suboptimal to model
    them separately. If we had more post-target fixations it probably made sense."""
    mask_onset = evts[evts['trigger']==94].index.values
    mask_offset = evts[evts['trigger']==95].index.values
    
    mask_target = evts[evts['trigger']==999].index.values
    
    trial_times = list(zip(mask_onset, mask_offset))
    for trial in trial_times:
        if any(evts['trigger'].iloc[trial[0] : trial[1]+1].isin([991, 992, 993, 994])):
            targ = evts.iloc[trial[0] : trial[1]+1][evts['trigger'].iloc[trial[0] : trial[1]+1].isin([991, 992, 993, 994])].index.values[0]
            evts['trigger'].iloc[trial[0] : targ] = evts['trigger'].iloc[trial[0] : targ].apply(lambda x: 905 if x==901 else x)
            evts['trigger'].iloc[targ+1 : trial[1]+1] = evts['trigger'].iloc[targ+1 : trial[1]+1].apply(lambda x: 907 if x==901 else x)
        else:
            evts['trigger'].iloc[trial[0] : targ] = evts['trigger'].iloc[trial[0] : trial[1]+1].apply(lambda x: 905 if x==901 else x)
    return evts

def prepare_data_for_unfold(sbj_id):
    meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv', header=0)
    
    def ovr_sub(ovr):
        if ovr in ['nover', 'novr', 'novrw']:
            ovr = ''
        elif ovr in ['ovrw', 'ovr', 'over', 'overw']:
            ovr = '_ovrw'
        elif ovr in ['ovrwonset', 'ovrons', 'overonset']:
            ovr = '_ovrwonset'
        return ovr
    
    
    subject = str(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    raw = mne.io.read_raw(path.join(sbj_path, "block1_sss_f_raw.fif"))
    t0 = raw.first_samp
    
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)    
    
    raw.info['bads'] = bad_eeg
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
    
    all_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_all_events.fif'))
    
    all_evts = pd.DataFrame(all_evts, columns=['time','useless','trigger'])
    
    fixations = all_evts[all_evts['trigger']==901]
    targ_n_fix = pd.concat([pd.DataFrame(target_evts, columns=['time','useless','trigger']), fixations])
    targ_n_fix = targ_n_fix.sort_values(by=['time'])
    targ_n_fix = pd.concat([pd.DataFrame(target_evts, columns=['time','useless','trigger']), fixations])
    targ_n_fix = targ_n_fix.drop_duplicates(subset=['time'], keep='first')
    targ_n_fix['trigger'].value_counts()
    targ_n_fix = targ_n_fix.sort_values(by=['time'])
    targ_n_fix['time'] = targ_n_fix['time'] - t0
    
    data_ds = raw.copy().resample(250)

    eeg = data_ds.get_data(picks=['eeg'])
    meg = data_ds.get_data(picks=['meg'])
    
    t_ds = targ_n_fix.copy()
    t_ds['time'] = (t_ds['time']/4).apply(np.floor).astype(int) # careful this is must be same sampling rate of data
    ev = t_ds[t_ds['trigger'].isin([991,992,993,994,901,999])].reset_index(drop=True)
    ev
    ev['trigger'].value_counts()
    ev = ev.rename(columns={'time': 'latency'})
    ev['intercept'] = 1
    ev['type'] = 'fixation'
    ev['type'].loc[ev['trigger']==999] = 'target'
    ev['type'].loc[ev['trigger'].isin([991,992,993,994])] = 'target'
    
    # enc = OneHotEncoder(handle_unknown='ignore')
    # enc_df = pd.DataFrame(enc.fit_transform(ev[['trigger']]).toarray())
    
    # evts_dummy = ev.join(enc_df)
    # evts_dummy = evts_dummy.drop(['useless', 0], axis=1)
    # evts_dummy = evts_dummy.rename(columns={1:"AP", 2:"CP", 3:"AU", 4:"CU"})
    # evts_dummy.to_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/evts_sbj_{subject}_correct.csv", index=False)
    
    # evts_dummy = ev.join(enc_df)
    ev = ev.drop(['useless'], axis=1)
    
    ev_pred = ev.copy()
    ev_pred['Predictability'] = np.nan
    ev_pred['Predictability'].loc[ev_pred['trigger'].isin([991,992])] = "Predictable"
    ev_pred['Predictability'].loc[ev_pred['trigger'].isin([993,994])] = "Unpredictable"
    
    ev_pred['Concreteness'] = np.nan
    ev_pred['Concreteness'].loc[ev_pred['trigger'].isin([991,993])] = "Abstract"
    ev_pred['Concreteness'].loc[ev_pred['trigger'].isin([992,994])] = "Concrete"
    
    ev_pred.to_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/evts_sbj_{subject}_concpred.csv", index=False)
    
    h5f = h5py.File(f'/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/EMEG_data_sbj_{subject}.h5', 'w')
    h5f.create_dataset('eeg', data=eeg)
    h5f.create_dataset('meg', data=meg)
    h5f.close()



if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
                21,22,23,24,25,26,27,28,29,30]
else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    prepare_data_for_unfold(ss)    