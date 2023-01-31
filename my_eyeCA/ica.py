#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 12:10:45 2022

@author: py01
"""

import mne
from mne import preprocessing as mp
from my_eyeCA import preprocess as pp
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

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

def run_ica_pipeline(raw, evt, method, cov_estimator, n_comp, drf=None):

    # Handle folder/file management
    try:
        fpath = Path(raw.filenames[0])    
        fpath = Path(drf)
        fname = fpath.stem
        tag = f"{fname}_ICA_{method}_{n_comp}_COV_{cov_estimator}"
        out_dir = fpath.parent / "ICA" / tag
        out_dir.mkdir(exist_ok=True, parents=True)
    except:
        fpath = Path(drf)
        fname = fpath.stem
        tag = f"{fname}_ICA_{method}_{n_comp}_COV_{cov_estimator}"
        out_dir = fpath.parent / "ICA_ovr_w" / tag
        out_dir.mkdir(exist_ok=True, parents=True)

    # Take subset of events for this instance of Raw
    #evt = pp.subset_events(raw, evt)

    # Filter and downsample
    draw, devt = pp.downsample_and_filter(raw, evt, lf=2, hf=40, sf=200)

       # %%
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