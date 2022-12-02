#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:58:21 2022

@author: fm02
adapted from py01 EyeCA
"""

import sys
import os
from os import path

import numpy as np
import pandas as pd

import mne
from mne.preprocessing import ICA, create_eog_epochs
import seaborn as sns
import matplotlib.pyplot as plt


os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

mne.viz.set_browser_backend("matplotlib")


reject_criteria = config.epo_reject
flat_criteria = config.epo_flat


# %%
# Set MNE's log level to DEBUG
mne.set_log_level(verbose="DEBUG")

sbj_id = int(sys.argv[1])

sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

# raw-filename mappings for this subject
tmp_fnames = config.sss_map_fnames[sbj_id][1]

# only use files for correct conditions
sss_map_fnames = []
for sss_file in tmp_fnames:
    sss_map_fnames.append(sss_file)

data_raw_files = []

# load unfiltered data to fit ICA with
for raw_stem_in in sss_map_fnames:
    data_raw_files.append(
        path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_ica_raw.fif'))

bad_eeg = config.bad_channels[sbj_id]['eeg']


	# %%
data = []
for drf in data_raw_files:
    data.append(mne.io.read_raw_fif(drf, preload=True))

print('Concatenating data')
data = mne.concatenate_raws(data)

data.info['bads'] = bad_eeg

evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_target_events.fif')   
devents = mne.read_events(evt_file)

event_dict = {'FRP': 999}
epochs = mne.Epochs(data, devents, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                reject=reject_criteria, flat=flat_criteria,
                preload=True)
evoked = epochs['FRP'].average()

FRP_fig = evoked.plot_joint(times=[0, .110, .167, .210, .266, .330, .430])

for i, fig in zip(['EEG','MAG','GRAD'], FRP_fig):
    fname_fig = path.join(sbj_path, 'Figures', f'FRP_all_{i}_ica_perblock_raw.png')
    fig.savefig(fname_fig)     
    
rows = np.where(devents[:,2]==999)
for row in rows[0]:
    if devents[row-2, 2] == 1:
        devents[row, 2] = 991
    elif devents[row-2, 2] == 2:
        devents[row, 2] = 992
    elif devents[row-2, 2] == 3:
        devents[row, 2] = 993
    elif devents[row-2, 2] == 4:
        devents[row, 2] = 994
    elif devents[row-2, 2] == 5:
        devents[row, 2] = 995
        
event_dict = {'Abstract/Predictable': 991, 
              'Concrete/Predictable': 992,
              'Abstract/Unpredictable': 993, 
              'Concrete/Unpredictable': 994}

epochs = mne.Epochs(data, devents, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                    reject=reject_criteria, flat=flat_criteria,
                    preload=True)
  
cond1 = 'Predictable'
cond2 = 'Unpredictable'

fig, ax = plt.subplots()
params = dict(spatial_colors=True, #show=False,
              time_unit='s')
epochs[cond1].average().plot(**params)
fname_fig = path.join(sbj_path, 'Figures', 'FRP_predictable_ica_perblock_raw.png')
plt.savefig(fname_fig)
epochs[cond2].average().plot(**params)
fname_fig = path.join(sbj_path, 'Figures', 'FRP_unpredictable_ica_perblock_raw.png')
plt.savefig(fname_fig)
contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()],
                              weights=[1, -1])
contrast.plot(**params)
fname_fig = path.join(sbj_path, 'Figures', 'FRP_predictability_ica_perblock_raw.png')
plt.savefig(fname_fig)

cond1 = 'Concrete'
cond2 = 'Abstract'

fig, ax = plt.subplots()
params = dict(spatial_colors=True, show=False,
              time_unit='s')
epochs[cond1].average().plot(**params)
fname_fig = path.join(sbj_path, 'Figures', 'FRP_concrete_ica_perblock_raw.png')
plt.savefig(fname_fig)
epochs[cond2].average().plot(**params)
fname_fig = path.join(sbj_path, 'Figures', 'FRP_abstract_ica_perblock_raw.png')
plt.savefig(fname_fig)
contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()],
                              weights=[1, -1])
contrast.plot(**params)
fname_fig = path.join(sbj_path, 'Figures', 'FRP_concreteness_ica_perblock_raw.png')
plt.savefig(fname_fig)







