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

def ovr_sub(ovr):
    if ovr in ['nover', 'novr', 'novrw']:
        ovr = ''
    elif ovr in ['ovrw', 'ovr', 'over', 'overw']:
        ovr = '_ovrw'
    elif ovr in ['ovrwonset', 'ovrons', 'overonset']:
        ovr = '_ovrwonset'
    return ovr

def compute_covariance_from_raw(sbj_id, cov_method='empirical', save_covmat=False, plot_covmat=False):
    
    subject = str(sbj_id)
    
    ovr = config.ovr_procedure[sbj_id]
    
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw_test = apply_ica.get_ica_raw(sbj_id, 
                                     condition='both',
                                     overweighting=ovr,
                                     interpolate=False, 
                                     drop_EEG_4_8=False)
    
    raw_test = raw_test.set_eeg_reference(ref_channels='average', projection=True)
    raw_test.load_data()
    raw_test.info['bads'] = bad_eeg
   
    # raw_test.drop_channels(['EEG004', 'EEG008'])
    # raw_test.interpolate_bads(reset_bads=True)
    
    # with this script we want to drop all bad channels for covariance computation
    # and we will do the same when computing evokeds
    picks = mne.pick_types(raw_test.info, meg=True, eeg=True, exclude='bads')

    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
 
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
    
    evt.apply(lambda x: x['time']+int(34*1e-3*raw_test.info['sfreq']) if x['trigger'] in [93, 94, 95] else x['time'], axis=1)
    
    event_dict = {'Stim_on': 94}
    
    epochs = mne.Epochs(raw_test, evt, tmin=-0.350, tmax=-0.150, event_id=event_dict,
                        baseline=(-0.350, -0.150),
                        flat=None, picks=picks, reject_by_annotation=False, 
                        reject=None, preload=True)
    if cov_method==['auto']:
        noise_cov=mne.compute_covariance(epochs, method='auto', 
                                                 tmax=-0.150, rank='info', return_estimators=True)
    else:
        noise_cov = list()
        for cov in cov_method:
            noise_cov.append(mne.compute_covariance(epochs, method=cov, 
                                                     tmax=-0.150, rank='info', return_estimators=True))
    
    # the bits below regularise the covariance matrix if computed using empirical
    # HOWEVER, MNE document suggests to do it the compute_covariance function
    # ALSO diagonal_fixed is equivalent to regularising and empirical covariance
    
    # if cov_method==['empirical']:
    #     noise_cov = mne.cov.regularize(noise_cov, epochs.info, mag=0.1, grad=0.1,
    #                                eeg=0.1, rank='info')
    # elif len(noise_cov)>1:
    #     for i_nc, nc in enumerate(noise_cov):
    #         if nc['method']=='empirical':
    #             noise_cov[i_nc] = mne.cov.regularize(nc, epochs.info, mag=0.1, grad=0.1,
    #                                        eeg=0.1, rank='info')
                
    if save_covmat:
        if len(noise_cov)==1:
            fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  f"_covariancematrix_{cov_method[0]}_dropbads-cov.fif")
            mne.write_cov(fname_cov, noise_cov[0], overwrite=True)
        elif len(noise_cov)>1:
            for nc in noise_cov:
                fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                      f"_covariancematrix_{nc['method']}_dropbads-cov.fif")
                mne.write_cov(fname_cov, nc, overwrite=True)
                
    if plot_covmat:
        if len(cov_method)==1:
            if cov_method=='auto':
                figs = noise_cov[0].plot(epochs.info, proj=True)
            else:
                figs = noise_cov.plot(epochs.info, proj=True)
            
            for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
                fname_fig = path.join(sbj_path, 'Figures', f'covariance_{cov_method[0]}_dropbads_{i}.png')
                fig.savefig(fname_fig)
        
            evoked = epochs.average()
            fig = evoked.plot_white(noise_cov, time_unit='s')
            fname_fig = path.join(sbj_path, 'Figures', f'whitened_cov_{cov_method[0]}_dropbads.png')
            fig.savefig(fname_fig)
                
        elif len(cov_method)>1:
            for nc in noise_cov:
                figs = nc.plot(epochs.info, proj=True)
                
                for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
                    fname_fig = path.join(sbj_path, 'Figures', f'covariance_{nc["method"]}_dropbads_{i}.png')
                    fig.savefig(fname_fig)
        
        
                evoked = epochs.average()
                fig = evoked.plot_white(nc, time_unit='s')
                fname_fig = path.join(sbj_path, 'Figures', f'whitened_cov_{nc["method"]}_dropbads.png')
                fig.savefig(fname_fig)


def compute_covariance_from_ICA_raw(sbj_id, cov_method=['empirical'], save_covmat=False, plot_covmat=False):
    
    assert isinstance(cov_method, list)
    
    subject = str(sbj_id)
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)        
    raw.info['bads'] = bad_eeg

    
    # with this script we want to drop all bad channels for covariance computation
    # and we will do the same when computing evokeds
    if sbj_id==12:
        picks = mne.pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    else:
        picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')

    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
 
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
    
    evt.apply(lambda x: x['time']+int(34*1e-3*raw.info['sfreq']) if x['trigger'] in [93, 94, 95] else x['time'], axis=1)
    
    event_dict = {'Stim_on': 94}
    
    epochs = mne.Epochs(raw, evt, tmin=-0.350, tmax=-0.150, event_id=event_dict,
                        baseline=(-0.350, -0.150),
                        flat=None, picks=picks, reject_by_annotation=False, 
                        reject=None, preload=True)
    if cov_method==['auto']:
        noise_cov=mne.compute_covariance(epochs, method='auto', 
                                                 tmax=-0.150, rank='info', return_estimators=True)
    else:
        noise_cov = list()
        for cov in cov_method:
            noise_cov.append(mne.compute_covariance(epochs, method=cov, 
                                                     tmax=-0.150, rank='info', return_estimators=True))
            
    # if cov_method==['empirical']:
    #     noise_cov = mne.cov.regularize(noise_cov, epochs.info, mag=0.1, grad=0.1,
    #                                eeg=0.1, rank='info')
    # elif len(cov_method)>1:
    #     for i_nc, nc in enumerate(noise_cov):
    #         if nc['method']=='empirical':
    #             noise_cov[i_nc] = mne.cov.regularize(nc, epochs.info, mag=0.1, grad=0.1,
    #                                        eeg=0.1, rank='info')
                
    if save_covmat:
        if len(cov_method)==1:
            fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  f"_covariancematrix_{cov_method[0]}_dropbads-cov.fif")
            mne.write_cov(fname_cov, noise_cov[0], overwrite=True)
        elif len(cov_method)>1:
            for nc in noise_cov:
                fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                      f"_covariancematrix_{nc['method']}_dropbads-cov.fif")
                mne.write_cov(fname_cov, nc, overwrite=True)
                
    if plot_covmat:
        if len(cov_method)==1:
            figs = noise_cov[0].plot(epochs.info, proj=True)
            
            for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
                fname_fig = path.join(sbj_path, 'Figures', f'covariance_{cov_method[0]}_dropbads_{i}.png')
                fig.savefig(fname_fig)
        
            evoked = epochs.average()
            fig = evoked.plot_white(noise_cov, time_unit='s')
            fname_fig = path.join(sbj_path, 'Figures', f'whitened_cov_{cov_method[0]}_dropbads.png')
            fig.savefig(fname_fig)
                
        elif len(cov_method)>1:
            for nc in noise_cov:
                figs = nc.plot(epochs.info, proj=True)
                
                for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
                    fname_fig = path.join(sbj_path, 'Figures', f'covariance_{nc["method"]}_dropbads_{i}.png')
                    fig.savefig(fname_fig)
        
        
                evoked = epochs.average()
                fig = evoked.plot_white(nc, time_unit='s')
                fname_fig = path.join(sbj_path, 'Figures', f'whitened_cov_{nc["method"]}_dropbads.png')
                fig.savefig(fname_fig)


def compute_covariance_MEGonly_from_ICA_raw(sbj_id, cov_method=['empirical'], save_covmat=False, plot_covmat=False):
    
    assert isinstance(cov_method, list)
    
    subject = str(sbj_id)
    
    ovr = config.ovr_procedure[sbj_id]
    ovr = ovr_sub(ovr)
    
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    bad_eeg = config.bad_channels_all[sbj_id]['eeg']
    
    raw = list()
    for i in range(1,6):
        fpath = path.join(sbj_path, f'block{i}_sss_f_ica{ovr}_both_raw.fif')
        raw_block = mne.io.read_raw(fpath)
        raw.append(raw_block)
    
    raw = mne.concatenate_raws(raw, preload=True)        
    raw.info['bads'] = bad_eeg


    picks = mne.pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_target_events.fif'))
 
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
    
    evt.apply(lambda x: x['time']+int(34*1e-3*raw.info['sfreq']) if x['trigger'] in [93, 94, 95] else x['time'], axis=1)
    
    event_dict = {'Stim_on': 94}
    
    epochs = mne.Epochs(raw, evt, tmin=-0.350, tmax=-0.150, event_id=event_dict,
                        baseline=(-0.350, -0.150),
                        flat=None, picks=picks, reject_by_annotation=False, 
                        reject=None, preload=True)
    
    if cov_method==['auto']:
        noise_cov=mne.compute_covariance(epochs, method='auto', 
                                                 tmax=-0.150, rank='info', return_estimators=True)
    else:
        noise_cov = list()
        for cov in cov_method:
            noise_cov.append(mne.compute_covariance(epochs, method=cov, 
                                                     tmax=-0.150, rank='info', return_estimators=True))
            
    # if cov_method==['empirical']:
    #     noise_cov = mne.cov.regularize(noise_cov, epochs.info, mag=0.1, grad=0.1,
    #                                eeg=0.1, rank='info')
    # elif len(cov_method)>1:
    #     for i_nc, nc in enumerate(noise_cov):
    #         if nc['method']=='empirical':
    #             noise_cov[i_nc] = mne.cov.regularize(nc, epochs.info, mag=0.1, grad=0.1,
    #                                        eeg=0.1, rank='info')
                
    if save_covmat:
        if len(cov_method)==1:
            fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  f"_covariancematrix_MEGonly_{cov_method[0]}_dropbads-cov.fif")
            mne.write_cov(fname_cov, noise_cov[0], overwrite=True)
        elif len(cov_method)>1:
            for nc in noise_cov:
                fname_cov = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                      f"_covariancematrix_MEGonly_{nc['method']}_dropbads-cov.fif")
                mne.write_cov(fname_cov, nc, overwrite=True)
                
    if plot_covmat:
        if len(cov_method)==1:
            figs = noise_cov[0].plot(epochs.info, proj=True)
            
            for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
                fname_fig = path.join(sbj_path, 'Figures', f'covariance_MEGonly_{cov_method[0]}_dropbads_{i}.png')
                fig.savefig(fname_fig)
        
            evoked = epochs.average()
            fig = evoked.plot_white(noise_cov, time_unit='s')
            fname_fig = path.join(sbj_path, 'Figures', f'whitened_cov_MEGonly_{cov_method[0]}_dropbads.png')
            fig.savefig(fname_fig)
                
        elif len(cov_method)>1:
            for nc in noise_cov:
                figs = nc.plot(epochs.info, proj=True)
                
                for i, fig in zip(['matrix', 'eigenvalue_index'], figs):
                    fname_fig = path.join(sbj_path, 'Figures', f'covariance_MEGonly_{nc["method"]}_dropbads_{i}.png')
                    fig.savefig(fname_fig)
        
        
                evoked = epochs.average()
                fig = evoked.plot_white(nc, time_unit='s')
                fname_fig = path.join(sbj_path, 'Figures', f'whitened_cov_{nc["method"]}_dropbads.png')
                fig.savefig(fname_fig)


# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]


# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     compute_covariance(ss)    
    
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
