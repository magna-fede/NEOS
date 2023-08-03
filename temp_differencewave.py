#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:11:48 2023

@author: fm02
"""

import pandas as pd
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt


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

sbj_id =1


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

raw.resample(250)

raw.pick_types(meg=True, eeg=True)

info = raw.info
a = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/effect_{sbj_id}_pred.csv")

target = a[['channel', 'Predictability','yhat', 'time']][a['basisname']=='targ']
fix = a[['channel', 'Predictability','yhat', 'time']][a['basisname']=='fix']

eeg_P = [fix['yhat'][(fix['Predictability']=='Predictable') & (fix['channel']==ch)] for ch in range(1,65)]

meg_P = [fix['yhat'][(fix['Predictability']=='Predictable') & (fix['channel']==ch)] for ch in range(65,371)]
eeg_P = np.array(eeg_P)
meg_P = np.array(meg_P)

data_P = np.concatenate(([eeg_P, meg_P]))
evoked_P = mne.EvokedArray(data_P, info, tmin=-0.15)

eeg_U = [fix['yhat'][(fix['Predictability']=='Unpredictable') & (fix['channel']==ch)] for ch in range(1,65)]

meg_U = [fix['yhat'][(fix['Predictability']=='Unpredictable') & (fix['channel']==ch)] for ch in range(65,371)]
eeg_U = np.array(eeg_U)
meg_U = np.array(meg_U)

data_U = np.concatenate(([eeg_U, meg_U]))
evoked_U = mne.EvokedArray(data_U, info, tmin=-0.15)

contrast = mne.combine_evoked([evoked_P, evoked_U], weights=[1, -1])


contrast.plot_joint()
eeg_P = [target['yhat'][(target['Predictability']=='Predictable') & (target['channel']==ch)] for ch in range(1,65)]

meg_P = [target['yhat'][(target['Predictability']=='Predictable') & (target['channel']==ch)] for ch in range(65,371)]
eeg_P = np.array(eeg_P)
meg_P = np.array(meg_P)

data_P = np.concatenate(([eeg_P, meg_P]))
evoked_P = mne.EvokedArray(data_P, info, tmin=-0.15)

eeg_U = [target['yhat'][(target['Predictability']=='Unpredictable') & (target['channel']==ch)] for ch in range(1,65)]

meg_U = [target['yhat'][(target['Predictability']=='Unpredictable') & (target['channel']==ch)] for ch in range(65,371)]
eeg_U = np.array(eeg_U)
meg_U = np.array(meg_U)

data_U = np.concatenate(([eeg_U, meg_U]))
evoked_U = mne.EvokedArray(data_U, info, tmin=-0.15)

contrast = mne.combine_evoked([evoked_P, evoked_U], weights=[1, -1])


contrast.plot_joint()