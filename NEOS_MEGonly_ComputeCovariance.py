#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:21:50 2023

@author: fm02
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

#os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import preprocess, ica, snr_metrics, apply_ica

os.chdir("/home/fm02/MEG_NEOS/NEOS")

mne.viz.set_browser_backend("matplotlib")

# %%

cov_method = 'auto'

def compute_covariance(sbj_id):

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
    
    picks = mne.pick_types(raw_test.info, meg=True, eeg=False, exclude='bads')

    # manually checked for the shortest possible time the fixation cross was on
    # this is 
    evt = pd.DataFrame(target_evts, columns=['time','n','trigger'])
    
    # we need to consider 34ms delay for those triggers that reflect a change in the display
    # note that we don't need to add the delay for the eye events (fixation/saccades)
    # (i.e., they are not generated by a change on screen, so they are not affected by this delay)
    # the events that are affected are:
    #    when fixation cross appears (TRIGGER 93)
    #    when the sentence appears (TRIGGER 94)
    #    when the sentence disappears (TRIGGER 95)
    #    when the calibration screen is presented (we ignore that by aligning the triggers)
    # alternatively we might ignore this problem and get the stimuli even before, it should
    # not matter for covariance matrix
    
    evt.apply(lambda x: x['time']+34 if x['trigger'] in [93, 94,95] else x['time'], axis=1)
    
    event_dict = {'Stim_on': 94}
    
    epochs = mne.Epochs(raw_test, evt, tmin=-0.450, tmax=0.0, event_id=event_dict,
                        picks=picks, flat=None,
                        reject_by_annotation=False, reject=None, preload=True)

    noise_cov_auto = mne.compute_covariance(epochs, method=cov_method,
                                                tmax=0, rank='info', return_estimators=True)
    winner = mne.Info(noise_cov_auto[0])['method']
    
    fname = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              'MEGonly_covariancematrix_auto-cov.fif')
    
    mne.write_cov(fname, noise_cov_auto[0])
    
    figs = noise_cov_auto[0].plot(raw_test.info, proj=True)

    for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
        fname_fig = path.join(sbj_path, 'Figures', f'MEGonly_covariance_{winner}_{i}.png')
        fig.savefig(fname_fig)



if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
               21,22,23,24,25,26,27,28,29,30]


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    compute_covariance(ss)    
    
# this was used to check the epoch duration, keeping it for record
# and in case it's useful in the future     
    # fix_on = np.where(evt['trigger']==93)[0]
    # fix_off = np.where(evt['trigger']==94)[0]
             
    
    # for i in range(len(fix_off)):
    #     if fix_off[i]-fix_on[i] != 1:
    #         fix_on = np.delete(fix_on, i)
    
    # fix_times = tuple(zip(fix_on, fix_off)) 
   
    # fix_selection = dict.fromkeys(['data', 'time'])
    # fix_selection['data'] = list()
    # fix_selection['time'] = list()
    
    # for i, indices in enumerate(fix_times):
    #     # adding -20 ms prior to saccade onset and +10ms after saccade offset
    #     # 20 is fine as long as th
    #     d, t = raw[:,(evt.iloc[indices[0]][0] - t0) : \
    #                     (evt.iloc[indices[1]][0] - t0)]
    #     # mean centre each saccade epoch
    #     d -= d.mean(axis=1).reshape(-1,1)    
    #     fix_selection['data'].append(d)    
    #     fix_selection['time'].append(t) 



    # fix_concatenated = np.concatenate(fix_selection['data'], axis=1)
    
    # num.append(pd.Series([trial.shape[1] for trial in fix_selection['data']]))
        
        # transform to Raw object
    # data_for_cov = mne.io.BaseRaw(info=raw.info, preload=fix_concatenated)
