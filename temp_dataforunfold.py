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
from sklearn.preprocessing import StandardScaler

import mne
os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
from my_eyeCA import apply_ica
from sklearn.preprocessing import OneHotEncoder
import h5py

sbj_id = 5

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
bad_eeg = config.bad_channels_all[sbj_id]['eeg']

ovr = config.ovr_procedure[sbj_id]
ovr = ovr_sub(ovr)

raw = mne.io.read_raw(path.join(sbj_path, "block1_sss_raw.fif"))
t0 = raw.first_samp

raw_test = apply_ica.get_ica_raw(sbj_id, 
                                 condition='both',
                                 overweighting=ovr,
                                 interpolate=False, 
                                 drop_EEG_4_8=False)

raw_test = raw_test.set_eeg_reference(ref_channels='average', projection=True)
raw_test.load_data()
raw_test.info['bads'] = bad_eeg

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

data_ds = raw_test.copy().resample(250)
d = data_ds.get_data(picks=['eeg','meg'])

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

enc = OneHotEncoder(handle_unknown='ignore')
enc_df = pd.DataFrame(enc.fit_transform(ev[['trigger']]).toarray())

evts_dummy = ev.join(enc_df)
evts_dummy = evts_dummy.drop(['useless', 0], axis=1)
evts_dummy = evts_dummy.rename(columns={1:"AP", 2:"CP", 3:"AU", 4:"CU"})
evts_dummy.to_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/evts_sbj_{subject}_correct.csv", index=False)


h5f = h5py.File(f'/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/data_sbj_{subject}.h5', 'w')
h5f.create_dataset('dataset_1', data=d)
h5f.close()
