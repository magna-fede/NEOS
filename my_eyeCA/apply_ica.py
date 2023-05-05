#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:10:45 2022

@author: py01
"""
from os import path
import mne
from mne import preprocessing as mp
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib

import NEOS_config as config
from my_eyeCA import preprocess as pp


reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

VALID_STATUS = {'', '_ovrw', '_ovrwonset'}
 
def over_input(status):
    if status not in VALID_STATUS:
        raise ValueError("results: status must be one of %r." % VALID_STATUS)

def apply_ica_pipeline(raw, evt, thresh=1.1, method='both', ica_filename=None, ica_instance=None, over=False, plot_overlay=False,
                       overwrite_saved=False):
    over_input(over)
    # Handle folder/file management
    fpath = Path(raw.filenames[0])

    t0 = raw.first_samp
    raw.load_data()
    
    if ica_filename:
        ic = mp.read_ica(ica_filename + ".fif")
    elif ica_instance:
        ic = ica_instance
        if over == '_ovrw':
            ica_filename = path.join(fpath.parent, 'ICA_ovr_w',
                                     f'{fpath.stem}_ICA_extinfomax_0.99_COV_None',
                                     f'{fpath.stem}_ICA_extinfomax_0.99_COV_None-ica')
        elif over == '_ovrwonset':
            ica_filename = path.join(fpath.parent, 'ICA_ovr_w_onset',
                             f'{fpath.stem}_ICA_extinfomax_0.99_COV_None',
                             f'{fpath.stem}_ICA_extinfomax_0.99_COV_None-ica')    
        elif over=='':            
            ica_filename = path.join(fpath.parent, 'ICA',
                                 f'{fpath.stem}_ICA_extinfomax_0.99_COV_None',
                                 f'{fpath.stem}_ICA_extinfomax_0.99_COV_None-ica')

    #evt = pp.subset_events(raw, evt)
    if method=='eog': 
        
        eog_idx, ic_scores = ic.find_bads_eog(raw, ch_name=["EOG001", "EOG002"])
        # fig = ic.plot_scores(scores, exclude=eog_idx, figsize=(16, 9), show=False)
        # fig.show()
        ic.exclude = eog_idx
        
        if plot_overlay:
            ic.plot_overlay(raw, eog_idx, picks='eeg')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_eeg.png'
            plt.savefig(fname_fig)
            
            ic.plot_overlay(raw, eog_idx, picks='mag')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_mag.png'
            plt.savefig(fname_fig)
            
            ic.plot_overlay(raw, eog_idx, picks='grad')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_grad.png'
            plt.savefig(fname_fig)    

        
        ### hacky solution for now 
        bads, raw.info['bads'] = raw.info['bads'], []
        
        # generate_report(ic, ic_scores, raw, ica_filename, config.reject)
        
        ic.apply(raw)

        if overwrite_saved:
            raw.save(fpath.parent / f"{fpath.stem[:-4]}_ica{over}_eog_raw.fif",
                    overwrite=True)   
            # Save fitted ICA
            ic_file = ica_filename + 'fitted_eog.fif'
            
            ic.save(ic_file, overwrite=True)
            
            ### revert back 
        raw.info['bads'] = bads
    
        return raw, ic, ic_scores
        
    elif method=='variance':
        pd_evt = pd.DataFrame(evt, columns=['time', 'previous', 'trigger'])
        pd_evt['time'] = pd_evt['time']*1e-3*raw.info['sfreq']
        # time is in samples right now        
        # convert it to seconds
        # this is useful for ICA components timecourse
        pd_evt['time'] = (pd_evt['time']-t0)/raw.info['sfreq']
    
        start_saccades = np.array(pd_evt['time'][pd_evt['trigger']==801])
        end_saccades = np.array(pd_evt['time'][pd_evt['trigger']==802])
        
        start_fixations = np.array(pd_evt['time'][pd_evt['trigger']==901])
        end_fixations = np.array(pd_evt['time'][pd_evt['trigger']==902])         
        
        times_sac = tuple(zip(start_saccades, end_saccades))
        
        times_fix = tuple(zip(start_fixations, end_fixations))

        print('Reading ICA file')
        components_timecourse = ic.get_sources(raw)
        
        var_sac = []
        
        for i, indices in enumerate(times_sac):
            section = components_timecourse.get_data(tmin=indices[0], tmax=indices[1])
            var_sac.append(np.var(section, axis=1))
        
        var_sac = np.dstack(var_sac).squeeze()
        var_sac = np.mean(var_sac, axis =1)
        
        
        var_fix = []
        
        for i, indices in enumerate(times_fix):
            section = components_timecourse.get_data(tmin=indices[0], tmax=indices[1])
            var_fix.append(np.var(section, axis=1))
        
        var_fix = np.dstack(var_fix).squeeze()
        var_fix = np.mean(var_fix, axis =1)
        
        ic_scores = (var_sac/var_fix)
        
        to_exclude = np.where(ic_scores > 1.1)[0]
        ic.exclude = to_exclude
        
        if plot_overlay:
            ic.plot_overlay(raw, to_exclude, picks='eeg')
            fname_fig =  Path(ica_filename).parent /  f'overlay{over}_{method}_eeg.png'
            plt.savefig(fname_fig)
            
            ic.plot_overlay(raw, to_exclude, picks='mag')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_mag.png'
            plt.savefig(fname_fig)
            
            ic.plot_overlay(raw, to_exclude, picks='grad')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_grad.png'
            plt.savefig(fname_fig)    

        
        ### hacky solution for now 
        bads, raw.info['bads'] = raw.info['bads'], []
        
        # generate_report(ic, ic_scores, raw, ica_filename, config.reject)
        
        ic.apply(raw)
        if overwrite_saved:            
            raw.save(fpath.parent / f"{fpath.stem[:-4]}_ica{over}_var_raw.fif",
                    overwrite=True)
                # Save fitted ICA
            ic_file = ica_filename + 'fitted_varcomp.fif'
            
            ic.save(ic_file, overwrite=True)
        ### revert back 
        raw.info['bads'] = bads
        
        return raw, ic, ic_scores
        
    
    elif method=='both':

        pd_evt = pd.DataFrame(evt, columns=['time', 'previous', 'trigger'])
        pd_evt['time'] = pd_evt['time']*1e-3*raw.info['sfreq']
        # time is in samples right now        
        # convert it to seconds
        # this is useful for ICA components timecourse
        pd_evt['time'] = (pd_evt['time']-t0)/raw.info['sfreq']
    
        start_saccades = np.array(pd_evt['time'][pd_evt['trigger']==801])
        end_saccades = np.array(pd_evt['time'][pd_evt['trigger']==802])
        
        start_fixations = np.array(pd_evt['time'][pd_evt['trigger']==901])
        end_fixations = np.array(pd_evt['time'][pd_evt['trigger']==902])         
        
        times_sac = tuple(zip(start_saccades, end_saccades))
        
        times_fix = tuple(zip(start_fixations, end_fixations))

        print('Reading ICA file')
        components_timecourse = ic.get_sources(raw)
        
        var_sac = []
        
        for i, indices in enumerate(times_sac):
            section = components_timecourse.get_data(tmin=indices[0], tmax=indices[1])
            var_sac.append(np.var(section, axis=1))
        
        var_sac = np.dstack(var_sac).squeeze()
        var_sac = np.mean(var_sac, axis =1)
        
        
        var_fix = []
        
        for i, indices in enumerate(times_fix):
            section = components_timecourse.get_data(tmin=indices[0], tmax=indices[1])
            var_fix.append(np.var(section, axis=1))
        
        var_fix = np.dstack(var_fix).squeeze()
        var_fix = np.mean(var_fix, axis =1)
        
        ic_scores = (var_sac/var_fix)
        
        to_exclude = np.where(ic_scores > 1.1)[0]
        
        eog_idx, ic_scores_eog = ic.find_bads_eog(raw, ch_name=["EOG001", "EOG002"])
        
        ic.exclude = list(set(eog_idx) | set(to_exclude))
        
        if plot_overlay:
            ic.plot_overlay(raw, list(set(eog_idx) | set(to_exclude)), picks='eeg')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_eeg.png'
            plt.savefig(fname_fig)
            
            ic.plot_overlay(raw, list(set(eog_idx) | set(to_exclude)), picks='mag')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_mag.png'
            plt.savefig(fname_fig)
            
            ic.plot_overlay(raw, list(set(eog_idx) | set(to_exclude)), picks='grad')
            fname_fig =  Path(ica_filename).parent / f'overlay{over}_{method}_grad.png'
            plt.savefig(fname_fig)    

        ### hacky solution for now 
        bads, raw.info['bads'] = raw.info['bads'], []
        
        # generate_report(ic, ic_scores, raw, ica_filename, config.reject)
        # generate_report(ic, ic_scores_eog, raw, ica_filename, config.reject)
        
        ic.apply(raw)
        
        if overwrite_saved:
            raw.save(fpath.parent / f"{fpath.stem[:-4]}_ica{over}_both_raw.fif",
                    overwrite=True)
                # Save fitted ICA
            ic_file = ica_filename + 'fitted_eogvar.fif'
            
            ic.save(ic_file, overwrite=True)
            
        ### revert back 
        raw.info['bads'] = bads
    
        return raw, ic, ic_scores



def plot_evoked_sensors(data, devents, comp_sel, all_factors=False, standard_rejection=True):
    
    # need to deal with downsampled data
    # adapt the events numpy array, check if this works
    loc_events = devents.copy()
    
    if data.info['sfreq']!=1000.0:
        loc_events[:,0] = loc_events[:,0]*1e-3*data.info['sfreq']
    
    fpath = Path(data.filenames[0])
    event_dict = {'FRP': 999}
    if standard_rejection:
        epochs = mne.Epochs(data, loc_events, picks=['meg', 'eeg'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                    reject=reject_criteria, flat=flat_criteria,
                    preload=True)
    else:
        epochs = mne.Epochs(data, loc_events, picks=['meg', 'eeg'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                    reject=None, flat=None, reject_by_annotation=False,
                    preload=True)    
        
    evoked = epochs['FRP'].average()
    
    FRP_fig = evoked.plot_joint(title= None,
                                times=[0, .110, .167, .210, .266, .330, .430])

    for i, fig in zip(['EEG','MAG','GRAD'], FRP_fig):
        fname_fig = fpath.parent / 'Figures' / f'FRP_all_{i}_{comp_sel}.png'
        fig.savefig(fname_fig)     
    
    if all_factors:
            
        rows = np.where(loc_events[:,2]==999)
        for row in rows[0]:
            if loc_events[row-2, 2] == 1:
                loc_events[row, 2] = 991
            elif loc_events[row-2, 2] == 2:
                loc_events[row, 2] = 992
            elif loc_events[row-2, 2] == 3:
                loc_events[row, 2] = 993
            elif loc_events[row-2, 2] == 4:
                loc_events[row, 2] = 994
            elif loc_events[row-2, 2] == 5:
                loc_events[row, 2] = 995
                
        event_dict = {'Abstract/Predictable': 991, 
                      'Concrete/Predictable': 992,
                      'Abstract/Unpredictable': 993, 
                      'Concrete/Unpredictable': 994}
        epochs = mne.Epochs(data, loc_events, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                            reject=reject_criteria, flat=flat_criteria,
                            preload=True)
        cond1 = 'Predictable'
        cond2 = 'Unpredictable'
        
        params = dict(title= None, spatial_colors=True, show=False,
                      time_unit='s')
        epochs[cond1].average().plot(**params)
        fname_fig =  fpath.parent / 'Figures' / f'FRP_predictable_ovrw_{comp_sel}.png'
        plt.savefig(fname_fig)
        epochs[cond2].average().plot(**params)
        fname_fig =  fpath.parent / 'Figures' / f'FRP_unpredictable_ovrw_{comp_sel}.png'
        plt.savefig(fname_fig)
        contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()],
                                      weights=[1, -1])
        contrast.plot(**params)
        fname_fig =  fpath.parent / 'Figures' / f'FRP_predictabibility_ovrw_{comp_sel}.png'
        plt.savefig(fname_fig)
        
        cond1 = 'Concrete'
        cond2 = 'Abstract'
        
        params = dict(title= None, spatial_colors=True, show=False,
                      time_unit='s')
        epochs[cond1].average().plot(**params)
        fname_fig = fpath.parent / 'Figures' / f'FRP_concrete_ovrw_{comp_sel}.png'
        plt.savefig(fname_fig)
        epochs[cond2].average().plot(**params)
        fname_fig =  fpath.parent / 'Figures' / f'FRP_abstract_ovrw_{comp_sel}.png'
        plt.savefig(fname_fig)
        contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()],
                                      weights=[1, -1])
        contrast.plot(**params)
        fname_fig =  fpath.parent / 'Figures' / f'FRP_concreteness_ovrw_{comp_sel}.png'
        plt.savefig(fname_fig)
        

# def generate_report(inst, ic_scores, raw, file_ica, reject):
    
#     report = mne.Report(title=file_ica.split('/')[-1])

#     # plot for specified channel types
#     for ch_type in ['eeg', 'mag', 'grad']:
#         fig_ic = inst.plot_components(ch_type=ch_type)
#         caption = [ch_type.upper() + ' Components' for i in fig_ic]
#         report.add_figure(fig_ic, title=ch_type.upper() +'Components', caption=caption,
#                                    section='ICA Components')

#     for eog_ch in ['EOG001', 'EOG002']:
#         # get single EOG trials

#         eog_epochs = mp.create_eog_epochs(raw, ch_name=eog_ch, reject=reject)
#         eog_average = eog_epochs.average()  # average EOG epochs

#         inds = inst.exclude

#         if inds != []:  # if some bad components found

#             fig_sc = inst.plot_scores(ic_scores, exclude=inds)
#             report.add_figure(fig_sc, caption=f'{eog_ch} Scores',
#                               title='Scores as var(saccade) / var(fixation)',
#                             section=f'{eog_ch} ICA component scores')

#             fig_rc = inst.plot_sources(raw)
#             report.add_figure(fig_rc, title='Sources', caption=f'{eog_ch} Sources',
#                               section=f'{eog_ch} raw ICA sources')

#             fig_so = inst.plot_sources(eog_average)
#             report.add_figure(fig_so, title='Raw EOG Sources', caption=f'{eog_ch} Sources',
#                               section=f'{eog_ch} ICA Sources')

#             fig_pr = inst.plot_properties(eog_epochs,  picks=inds,
#                                          psd_args={'fmax': 35.},
#                                          image_args={'sigma': 1.})

#             txt_str = f'{eog_ch} Properties'
#             caption = [txt_str for i in fig_pr]
#             report.add_figure(fig_pr, caption=caption, title='Properties',
#                                        section=f'{eog_ch} ICA Properties')

#             fig_ov = inst.plot_overlay(eog_average, exclude=inds)
#             report.add_figure(fig_ov, title='Overlay',
#                              caption=f'{eog_ch} Overlay',
#                              section=f'{eog_ch} ICA Overlay')
#             plt.close('all')
            
#     report.save(file_ica + '_varcomp.html', overwrite=True)



