#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script that mimicks OPTICAT steps.
Not running on raw data, but on epoched data (to avoid larger saccades
                                              at the end of the sentence
                                              driving ICA decopomposition.)
@author: federica.magnabosco@mrc-cbu.cam.ac.uk
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

reject = config.reject

mne.viz.set_browser_backend("matplotlib")


def run_ICA(sbj_id):
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    bad_eeg = config.bad_channels[sbj_id]['eeg']
    
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
            path.join(sbj_path, raw_stem_in[:-7] + 'sss_raw.fif'))
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf, preload=True))


    for block in data:
        block.filter(l_freq=2., h_freq=100., method='fir',
                    fir_design='firwin', filter_length='auto',
                    l_trans_bandwidth='auto', h_trans_bandwidth='auto')
        
    print('Concatenating data')
    data = mne.concatenate_raws(data)

    data.pick(picks=['eeg', 'meg', 'eog'])
    
    # it shouldn't matter whether eeg are average referenced
    # especially considering that ICA is fitted on ~70 components in our data
    # so there should be no problem of rank
    # plus MNE strongly advices to set reference as projection
    # not applying it for now (check if ICA aookies projections)
    
    print(f'Marking bad EEG channels: {bad_eeg}')
    data.info['bads'] = bad_eeg

    print('Interpolating bad EEG channels.')
    data.interpolate_bads(mode='accurate', reset_bads=True)

    print('Setting EEG reference.')
    data.set_eeg_reference(ref_channels='average', projection=True)

    all_events = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_all_events.fif'))   
    pd_all_events = pd.read_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  '_all_events_xy.csv'))
 
    t0 = data.first_samp
 
     
    # extract Raw Data
    raw_concatenated = data.get_data()
    
    
    ix_801 = np.where((pd_all_events['trigger']==801) & (pd_all_events['y'] < 700))[0]
    ix_802 = np.where((pd_all_events['trigger']==802) & (pd_all_events['y'] < 700))[0]
    
    sac_times = tuple(zip(ix_801, ix_802)) 
    
    sac_selection = dict.fromkeys(['data', 'time'])
    sac_selection['data'] = list()
    sac_selection['time'] = list()
    
    for i, indices in enumerate(sac_times):
        # adding -20 ms prior to saccade onset and +10ms after saccade offset
        # 20 is fine as long as th
        d, t = data[:,(all_events[indices[0]][0] - int(20*1e-3*data.info['sfreq']) - t0) : \
                        (all_events[indices[1]][0] + int(10*1e-3*data.info['sfreq']) - t0)]
        # mean centre each saccade epoch
        d -= d.mean(axis=1).reshape(-1,1)    
        sac_selection['data'].append(d)    
        sac_selection['time'].append(t) 

    sac_concatenated = np.concatenate(sac_selection['data'], axis=1)
    
    # same approach as in OPTICAT
    # repear saccade matrix until it is 0.5 at times as long as epochs
    sac_concatenated  = np.tile(sac_concatenated, int(np.ceil(raw_concatenated.shape[1]*0.5 / sac_concatenated.shape[1])))

    # and prune down until is exactly 0.5
    sac_concatenated = sac_concatenated[:, 0:int(raw_concatenated.shape[1]*0.5)]
    
    overweighted_for_ica = np.concatenate([raw_concatenated, sac_concatenated], axis=1)
    
    # transform to raw
    data_for_ica = mne.io.BaseRaw(info=data.info, preload=overweighted_for_ica)
    
    method = 'infomax'
    fit_params=dict(extended=True)
    n_components = 0.99
    decim = 3  # downsample data to save time

    # same random state for each ICA (not sure if beneficial?)
    random_state = 23
    
    EOG = ['EOG001','EOG002']
    # EOGthresh = 2.5 # try auto
        
    ica = ICA(n_components=n_components, method=method, fit_params=fit_params,
              random_state=random_state)        

    ica.fit(data_for_ica, decim=decim, reject=reject)
    report = mne.Report(subject=config.map_subjects[sbj_id][0],
                         title='ICA')
    
    for eog_ch in EOG:

        print('\n###\nFinding components for EOG channel %s.\n' % eog_ch)

        # get single EOG trials
        eog_epochs = create_eog_epochs(data, ch_name=eog_ch)
        
        if len(eog_epochs)>0:
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
        else:
            print('Something went wrong no EOG epochs detected')
    
    print(f'Saving ICA to {path.join(sbj_path, sbj_path[-3:] + "_sss_f_raw-ica_overweighted_unfiltered_raw.fif")}')
    ica.save(path.join(sbj_path, sbj_path[-3:] + '_sss_f_raw-ica_overweighted_unfiltered_raw.fif'), overwrite=True)
    report.save(path.join(sbj_path, 'Figures', 'report_ica_overweighted_unfiltered_raw.html'), overwrite=True)

    
    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    run_ICA(ss)        
        