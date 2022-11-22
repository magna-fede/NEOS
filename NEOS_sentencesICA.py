#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Epoch MEG data excluding saccades toward the right bottom corner.

author: federica.magnabosco@mrc-cbu.cam.ac.uk 
"""


import NEOS_config as config
import sys
import os
from os import path
import numpy as np
import pandas as pd

from importlib import reload
import pickle
import mne

from mne.preprocessing import ICA, create_eog_epochs
import seaborn as sns
import matplotlib.pyplot as plt

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

reload(config)

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat
mne.viz.set_browser_backend("matplotlib")


def run_ICA(sbj_id):
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    sbj_path_ET = path.join(
        config.path_ET, config.map_subjects[sbj_id][0][-3:])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # only use files for correct conditions
    sss_map_fnames = []
    for sss_file in tmp_fnames:
        sss_map_fnames.append(sss_file)

    data_raw_files = []

    for raw_stem_in in sss_map_fnames:
        data_raw_files.append(
            path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_raw.fif'))
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf))

    print('Concatenating data')
    data = mne.concatenate_raws(data)
    
    devents = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
        
    all_events = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_all_events.fif'))   
    pd_all_events = pd.read_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  '_all_events_xy.csv'))
    pd_trials = pd.read_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                   '_trial_synch.csv'))    
        
    t0 = data.first_samp
    
    ix_94 = np.where(devents ==94)[0]
    ix_95 = np.where(devents ==95)[0]

    startend = tuple(zip(ix_94, ix_95)) 
    
    t_max = pd_trials['meg_duration'].max()
    
    data_selection = dict.fromkeys(['data', 'time'])
    data_selection['data'] = list()
    data_selection['time'] = list()
    
    for i, indices in enumerate(startend):
        d, t = data[:,(devents[indices[0]][0] - t0) : (devents[indices[1]][0] -t0)]
        data_selection['data'].append(d)    
        data_selection['time'].append(t) 
    
    durations = [(devents[indices[1]][0] - t0) - (devents[indices[0]][0] -t0) for indices in startend]
    
    #add zero-padding each epoch at the end
    epochs_zeros = [np.concatenate([data_selection['data'][i], np.zeros((373, t_max-durations[i]))],axis=1) for i in range(400)]
    
    # epochs must be same length
    epochs_for_ica = mne.EpochsArray(epochs_zeros, info=data.info)

    # event_dict = {'Sentence': 94}

    # # don't baseline correct epochs
    # epochs = mne.Epochs(data, all_events, event_id=event_dict, tmin=0, tmax=t_max/1000,
    #                     baseline=None, preload=True)   
    
    # for i, epoch in enumerate(epochs):
    #     epochs[i].tmax=durations[i]
    
    method = 'infomax'
    n_components = 0.99
    decim = 3  # downsample data to save time

    # same random state for each ICA (not sure if beneficial?)
    random_state = 23
    
    EOG = ['EOG001','EOG002']
    # EOGthresh = 2.5 # try auto

    
    ChanTypes = ['eeg', 'meg']
    
    epochs_for_ica = epochs_for_ica.filter(l_freq=2., h_freq=None, fir_design='firwin').load_data()
        
    ica = ICA(n_components=n_components, method=method,
              random_state=random_state)        

    ica.fit(epochs_for_ica, decim=decim)
    report = mne.Report(subject=config.map_subjects[sbj_id][0],
                         title='ICA')
    data.load_data()
    for eog_ch in EOG:

        print('\n###\nFinding components for EOG channel %s.\n' % eog_ch)

        # get single EOG trials
        eog_epochs = create_eog_epochs(data, ch_name=eog_ch)

        eog_components, eog_scores = ica.find_bads_eog(inst=eog_epochs,
                                                       ch_name=eog_ch,  # a channel close to the eye
                                                       threshold='auto')  # lower than the default threshold
        ica.exclude = eog_components
        report.add_ica(
            ica=ica,
            title=f'ICA cleaning {eog_ch}',
            inst=data,
            eog_evoked=eog_epochs.average(),
            eog_scores=eog_scores,
            n_jobs=None  # could be increased!
        )
    
    print(f'Saving ICA to {path.join(sbj_path, sbj_path[-3:] + "_sss_f_raw-ica.fif")}')
    ica.save(path.join(sbj_path, sbj_path[-3:] + '_sss_f_raw-ica.fif'), overwrite=True)
    report.save(path.join(sbj_path, 'Figures', 'report_ica.html'), overwrite=True)

    
    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    run_ICA(ss)        
        