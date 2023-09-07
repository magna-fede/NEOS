#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 14:25:12 2023

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

mne.viz.set_browser_backend("matplotlib")

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
            21,22,23,24,25,26,27,28,29,30]

ave_path = path.join(config.data_path, "AVE")

predictables = list()
unpredictables = list()

concretes = list()
abstracts = list()

for sbj_id in sbj_ids:
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
    
    raw = mne.io.read_raw(path.join(sbj_path, f"block1_sss_f_ica{ovr}_both_raw.fif"))
    
    raw.resample(250)
    
    raw.pick_types(meg=True, eeg=True)
    
    info = raw.info
    eeg_file = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/eeg_effect_{sbj_id}_pred.csv")
    meg_file = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/meg_effect_{sbj_id}_pred.csv")
    
    target_eeg = eeg_file[['channel', 'Predictability','yhat', 'time']][eeg_file['basisname']=='targ']
    target_meg = meg_file[['channel', 'Predictability','yhat', 'time']][meg_file['basisname']=='targ']
    

    eeg_P = [target_eeg['yhat'][(target_eeg['Predictability']=='Predictable') & (target_eeg['channel']==ch)] \
              for ch in target_eeg['channel'].unique()]
    meg_P = [target_meg['yhat'][(target_meg['Predictability']=='Predictable') & (target_meg['channel']==ch)] \
              for ch in target_meg['channel'].unique()]
    
    eeg_P = np.array(eeg_P)
    meg_P = np.array(meg_P)
    
    data_P = np.concatenate(([eeg_P, meg_P]))
    evoked_P = mne.EvokedArray(data_P, info, tmin=-0.15)
    fig = evoked_P.plot_joint(times=[0, 0.11, 0.167, 0.21, 0.266, 0.33, 0.43])
    for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
        f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/{sbj_id}_{ch}_Predictable.png",
                  dpi=300)
        
    predictables.append(evoked_P)
    
    eeg_U = [target_eeg['yhat'][(target_eeg['Predictability']=='Unpredictable') & (target_eeg['channel']==ch)] \
              for ch in target_eeg['channel'].unique()]
    
    meg_U = [target_meg['yhat'][(target_meg['Predictability']=='Unpredictable') & (target_meg['channel']==ch)] \
              for ch in target_meg['channel'].unique()]
    eeg_U = np.array(eeg_U)
    meg_U = np.array(meg_U)
    
    data_U = np.concatenate(([eeg_U, meg_U]))
    evoked_U = mne.EvokedArray(data_U, info, tmin=-0.15)
    fig = evoked_U.plot_joint(times=[0, 0.11, 0.167, 0.21, 0.266, 0.33, 0.43])
    for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
        f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/{sbj_id}_{ch}_Unpredictable.png",
                  dpi=300)
        
    unpredictables.append(evoked_U)

    eeg_file = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/eeg_effect_{sbj_id}_conc.csv")
    meg_file = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/meg_effect_{sbj_id}_conc.csv")
    
    target_eeg = eeg_file[['channel', 'Concreteness','yhat', 'time']][eeg_file['basisname']=='targ']
    target_meg = meg_file[['channel', 'Concreteness','yhat', 'time']][meg_file['basisname']=='targ']
    

    eeg_C = [target_eeg['yhat'][(target_eeg['Concreteness']=='Concrete') & (target_eeg['channel']==ch)] \
              for ch in target_eeg['channel'].unique()]
    meg_C = [target_meg['yhat'][(target_meg['Concreteness']=='Concrete') & (target_meg['channel']==ch)] \
              for ch in target_meg['channel'].unique()]
    
    eeg_C = np.array(eeg_C)
    meg_C = np.array(meg_C)
    
    data_C = np.concatenate(([eeg_C, meg_C]))
    evoked_C = mne.EvokedArray(data_C, info, tmin=-0.15)
    fig = evoked_C.plot_joint(times=[0, 0.11, 0.167, 0.21, 0.266, 0.33, 0.43])
    for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
        f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/{sbj_id}_{ch}_Concrete.png",
                  dpi=300)
        
    concretes.append(evoked_C)
    
    eeg_A = [target_eeg['yhat'][(target_eeg['Concreteness']=='Abstract') & (target_eeg['channel']==ch)] \
              for ch in target_eeg['channel'].unique()]
    
    meg_A = [target_meg['yhat'][(target_meg['Concreteness']=='Abstract') & (target_meg['channel']==ch)] \
              for ch in target_meg['channel'].unique()]
    eeg_A = np.array(eeg_A)
    meg_A = np.array(meg_A)
    
    data_A = np.concatenate(([eeg_A, meg_A]))
    evoked_A = mne.EvokedArray(data_A, info, tmin=-0.15)
    fig = evoked_A.plot_joint(times=[0, 0.11, 0.167, 0.21, 0.266, 0.33, 0.43])
    for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
        f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/{sbj_id}_{ch}_Abstract.png",
                  dpi=300)
        
    abstracts.append(evoked_A)
    
[mne.write_evokeds(path.join(ave_path, f"{sbj_id}_Predictable_unfold_evoked-ave.fif"),
                  predictable) for sbj_id, predictable in zip(sbj_ids, predictables)]
[mne.write_evokeds(path.join(ave_path, f"{sbj_id}_Unpredictable_unfold_evoked-ave.fif"),
                  unpredictable) for sbj_id, unpredictable in zip(sbj_ids, unpredictables)]

[mne.write_evokeds(path.join(ave_path, f"{sbj_id}_Concrete_unfold_evoked-ave.fif"),
                  concrete) for sbj_id, concrete in zip(sbj_ids, concretes)]
[mne.write_evokeds(path.join(ave_path, f"{sbj_id}_Abstract_unfold_evoked-ave.fif"),
                  abstract) for sbj_id, abstract in zip(sbj_ids, abstracts)]

grand_average_P = mne.grand_average(predictables)
mne.write_evokeds(path.join(ave_path, "GA_predictable-ave.fif"),
                  grand_average_P)

grand_average_U = mne.grand_average(unpredictables)
mne.write_evokeds(path.join(ave_path, "GA_unpredictable-ave.fif"),
                  grand_average_U)

contrast = mne.combine_evoked([grand_average_U, grand_average_P], 
                              weights=[1, -1])
mne.write_evokeds(path.join(ave_path, "GA_predictability-contrast-ave.fif"),
                  contrast)

fig = grand_average_U.plot_joint()
for f, ch in zip(fig, ['EEG', 'GRAD', 'MAG']):
    f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/GA_{ch}_unpredictable.png",
              dpi=300)

fig = grand_average_P.plot_joint()
for f, ch in zip(fig, ['EEG', 'GRAD', 'MAG']):
    f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/GA_{ch}_predictable.png",
              dpi=300)

fig = contrast.plot_joint()
for f, ch in zip(fig, ['EEG', 'GRAD', 'MAG']):
    f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/GA_{ch}_predictability.png",
              dpi=300)

grand_average_C = mne.grand_average(concretes)
mne.write_evokeds(path.join(ave_path, "GA_concrete-ave.fif"),
                  grand_average_C)

grand_average_A = mne.grand_average(abstracts)
mne.write_evokeds(path.join(ave_path, "GA_abstract-ave.fif"),
                  grand_average_A)

contrast = mne.combine_evoked([grand_average_A, grand_average_C], 
                              weights=[1, -1])
mne.write_evokeds(path.join(ave_path, "GA_concreteness-contrast-ave.fif"),
                  contrast)

fig = grand_average_A.plot_joint()
for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
    f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/GA_{ch}_abstract.png",
              dpi=300)

fig = grand_average_C.plot_joint()
for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
    f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/GA_{ch}_concrete.png",
              dpi=300)

fig = contrast.plot_joint()
for f, ch in zip(fig, ['EEG', 'MAG', 'GRAD']):
    f.savefig(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/Figures/GA_{ch}_concreteness.png",
              dpi=300)
    