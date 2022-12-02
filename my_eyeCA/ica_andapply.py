#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:10:45 2022

@author: py01
"""

import mne
from mne import preprocessing as mp
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import NEOS_config as config
from my_eyeCA import preprocess as pp

def compute_noise_covariance(inst, time_win, method="shrunk"):

    cov = mne.compute_raw_covariance(
        inst,
        tmin=time_win[0],
        tmax=time_win[1],
        picks=["eeg", "meg"],
        method=method,
        # reject=dict(eeg=100e-6, eog=250e-6),
        tstep=2,
        rank="info",  # What is rank of data obtained from Maxwell filtering?
        n_jobs=-1,
    )

    return cov


def compute_ica(inst, cov, time_win, picks, method, n_comp=50):

    # Set up extended fitting parameters
    if method == "fastica":
        fitpars = None
        method = method
    elif method == "infomax":
        fitpars = dict(extended=False)
        method = method
    elif method == "extinfomax":
        fitpars = dict(extended=True)
        method = "infomax"

    # Initialize ICA object
    ica = mp.ICA(
        n_components=n_comp,
        noise_cov=cov,
        random_state=42,
        method=method,
        fit_params=fitpars,
        max_iter=1000,
        verbose="DEBUG",
    )

    ica.fit(inst, picks=picks, start=time_win[0], stop=time_win[1])

    return ica


def run_ica_pipeline(raw, evt, method, cov_estimator, n_comp):

    # Handle folder/file management
    fpath = Path(raw.filenames[0])
    fname = fpath.stem
    tag = f"{fname}_ICA_{method}_{n_comp}_COV_{cov_estimator}"
    out_dir = fpath.parent / "ICA" / tag
    out_dir.mkdir(exist_ok=True, parents=True)

    # Take subset of events for this instance of Raw
    evt = pp.subset_events(raw, evt)

    # Filter and downsample
    draw, devt = pp.downsample_and_filter(raw, evt, lf=2, hf=40, sf=200)

    draw.interpolate_bads(mode='accurate', reset_bads=True)


    # Compute noise covariance if requested, and generate plots
    if cov_estimator:

        # Get time limits for data to use for noise covariance
        dur = (devt[0, 0] - draw.first_samp) / draw.info["sfreq"]
        assert dur > 45, "Pre-first event duration less than 45s"

        tmin = 5  # 5 sec after onset of data
        tmax = round(dur - 5)  # 5 sec less than interval before first event

        # Compute covariance using chosen estimator
        cov = compute_noise_covariance(
            draw, time_win=[tmin, tmax], method=cov_estimator
        )

        # Save covariance matrix
        cov_file = out_dir / f"{tag}-cov.fif"
        cov.save(cov_file, overwrite=True)

        # Visualize
        fig_cov, fig_svd = cov.plot(draw.info, show=False)
        fig_cov.savefig(out_dir / f"{cov_file.stem}_covariance.png", dpi=150)
        plt.close(fig_cov)
        fig_svd.savefig(out_dir / f"{cov_file.stem}_svd.png", dpi=150)
        plt.close(fig_svd)

    else:
        cov = None

    # Compute ICA, and generate plots

    # Get limits of data to use for decomposition
    tmin = devt[0, 0] / draw.info["sfreq"] - 5  # 5 sec before first event
    tmax = devt[-1, 0] / draw.info["sfreq"] + 5  # 5 sec after last event

    # Fit ICA using given params
    ic = compute_ica(
        draw,
        cov=cov,
        time_win=[tmin, tmax],
        picks=["eeg", "meg"],
        method=method,
        n_comp=n_comp,
    )

    # Save fitted ICA
    ica_file = out_dir / f"{tag}-ica.fif"
    ic.save(ica_file, overwrite=True)

    # Plot first 20 components
    fig = ic.plot_components(
        picks=range(20), ch_type="eeg", show=False, title=tag
    )
    fig.savefig(out_dir / f"{ica_file.stem}_ics_topo.png", dpi=150)
    plt.close(fig)

    # Find components strongly correlated with EOG sensors and plot scores
    eog_idx, scores = ic.find_bads_eog(draw, ch_name=["EOG001", "EOG002"])
    fig = ic.plot_scores(
        scores, exclude=eog_idx, figsize=(16, 9), show=False, title=tag
    )
    fig.savefig(out_dir / f"{ica_file.stem}_eog_scores.png", dpi=150)
    plt.close(fig)

    # Plot topo of identified EOG components
    fig = ic.plot_components(
        picks=eog_idx, ch_type="eeg", show=False, title=tag
    )
    fig.savefig(out_dir / f"{ica_file.stem}_eog_ics_topo.png", dpi=150)
    plt.close(fig)

    # Plot time course of identified EOG components
    fig = ic.plot_sources(
        raw, picks=eog_idx, start=75, stop=90, show=False, title=tag
    )
    fig.savefig(out_dir / f"{ica_file.stem}_eog_ics_stc.png", dpi=150)
    plt.close(fig)

    # Return ICA object
    return ic

def apply_ica_pipeline(raw, evt, thresh, ica_filename):
    # Handle folder/file management
    fpath = Path(raw.filenames[0])

    t0 = raw.first_samp
    raw.load_data()

    evt = pp.subset_events(raw, evt)

    pd_evt = pd.DataFrame(evt, columns=['time', 'previous', 'trigger'])

    start_saccades = np.where(pd_evt['trigger']==801)[0]
    end_saccades = np.where(pd_evt['trigger']==802)[0] 

    start_fixations = np.where(pd_evt['trigger']==901)[0]
    end_fixations = np.where(pd_evt['trigger']==902)[0]
  

    times_sac = tuple(zip(start_saccades, end_saccades))

    sac_selection = dict.fromkeys(['data', 'time'])
    sac_selection['data'] = list()
    sac_selection['time'] = list()  

    for i, indices in enumerate(times_sac):

        d, t = raw[:,(evt[indices[0]][0] - t0) : (evt[indices[1]][0] - t0) ]

        sac_selection['data'].append(d)    
        sac_selection['time'].append(t) 


    times_fix = tuple(zip(start_fixations, end_fixations))

    fix_selection = dict.fromkeys(['data', 'time'])
    fix_selection['data'] = list()
    fix_selection['time'] = list()

    for i, indices in enumerate(times_fix):

        d, t = raw[:,(evt[indices[0]][0] - t0) : (evt[indices[1]][0] - t0) ]
        fix_selection['data'].append(d)    
        fix_selection['time'].append(t) 

        

    print('Reading ICA file')
    ic = mp.read_ica(ica_filename + ".fif")

    components_timecourse = ic.get_sources(raw)

    var_sac = []

    for event in sac_selection['time']:
        section = components_timecourse.get_data(tmin=event[0],
                                                 tmax=event[-1])
        var_sac.append(np.var(section, axis=1))

    var_sac = np.dstack(var_sac).squeeze()
    var_sac = np.mean(var_sac, axis =1)


    var_fix = []

    for event in fix_selection['time']:
        section = components_timecourse.get_data(tmin=event[0],
                                                 tmax=event[-1])
        var_fix.append(np.var(section, axis=1))

    var_fix = np.dstack(var_fix).squeeze()
    var_fix = np.mean(var_fix, axis =1)
    
    ic_scores = (var_sac/var_fix)

    to_exclude = np.where(ic_scores > 1.1)[0]
    ic.exclude = to_exclude
    
    ### hacky solution for now 
    bads, raw.info['bads'] = raw.info['bads'], []
    
    generate_report(ic, ic_scores, raw, ica_filename, config.reject)
    
    ic.apply(raw)
    raw.save(fpath.parent / f"{fpath.stem[:-4]}_ica_raw.fif",
            overwrite=True)

        # Save fitted ICA
    ic_file = ica_filename + '_varcomp.fif'

    ic.save(ic_file, overwrite=True)
    
    ### revert back 
    raw.info['bads'] = bads
    
    return raw, ic, ic_scores

def generate_report(inst, ic_scores, raw, file_ica, reject):
    
    report = mne.Report(title=file_ica.split('/')[-1])

    # plot for specified channel types
    for ch_type in ['eeg', 'mag', 'grad']:
        fig_ic = inst.plot_components(ch_type=ch_type)
        caption = [ch_type.upper() + ' Components' for i in fig_ic]
        report.add_figure(fig_ic, title=ch_type.upper() +'Components', caption=caption,
                                   section='ICA Components')

    for eog_ch in ['EOG001', 'EOG002']:
        # get single EOG trials

        eog_epochs = mp.create_eog_epochs(raw, ch_name=eog_ch, reject=reject)
        eog_average = eog_epochs.average()  # average EOG epochs

        inds = inst.exclude

        if inds != []:  # if some bad components found

            fig_sc = inst.plot_scores(ic_scores, exclude=inds)
            report.add_figure(fig_sc, caption=f'{eog_ch} Scores',
                              title='Scores as var(saccade) / var(fixation)',
                            section=f'{eog_ch} ICA component scores')

            fig_rc = inst.plot_sources(raw)
            report.add_figure(fig_rc, title='Sources', caption=f'{eog_ch} Sources',
                              section=f'{eog_ch} raw ICA sources')

            fig_so = inst.plot_sources(eog_average)
            report.add_figure(fig_so, title='Raw EOG Sources', caption=f'{eog_ch} Sources',
                              section=f'{eog_ch} ICA Sources')

            fig_pr = inst.plot_properties(eog_epochs,  picks=inds,
                                         psd_args={'fmax': 35.},
                                         image_args={'sigma': 1.})

            txt_str = f'{eog_ch} Properties'
            caption = [txt_str for i in fig_pr]
            report.add_figure(fig_pr, caption=caption, title='Properties',
                                       section=f'{eog_ch} ICA Properties')

            fig_ov = inst.plot_overlay(eog_average, exclude=inds)
            report.add_figure(fig_ov, title='Overlay',
                             caption=f'{eog_ch} Overlay',
                             section=f'{eog_ch} ICA Overlay')
            plt.close('all')
            
    report.save(file_ica + '_varcomp.html', overwrite=True)



