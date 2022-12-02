#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for preprocessing raw data: removing bad channels, downsampling,
filtering, epoching raw data and estimating noise covariance.

@author: py01
"""
from autoreject import Ransac
import mne


from autoreject import Ransac
import mne


def load_data(inst):
    if inst.preload:
        pass
    else:
        inst.load_data()


def subset_events(raw, evt):

    smin = raw.first_samp
    smax = raw.last_samp
    mask = (evt[:, 0] > smin) & (evt[:, 0] < smax)
    return evt[mask, :]


def downsample_and_filter(raw, evt, lf, hf, sf):

    # Load data
    load_data(raw)

    # High-pass
    d_raw = raw.filter(
        l_freq=lf,
        h_freq=None,
        picks=["eeg", "meg"],
        method="fir",
        phase="zero-double",
        n_jobs=-1,
    )

    # Resample
    if sf:
        d_raw, d_evt = d_raw.resample(sfreq=sf, events=evt, n_jobs=-1)
    else:
        d_evt = evt

    # Low-pass
    d_raw = d_raw.filter(
        l_freq=None,
        h_freq=hf,
        picks=["eeg", "meg"],
        method="fir",
        phase="zero-double",
        n_jobs=-1,
    )

    return d_raw, d_evt


def find_bad_channels(raw, evt):

    # Epoch data
    epo = mne.Epochs(
        raw,
        evt,
        picks="all",
        event_id=901,
        tmin=-0.5,
        tmax=1.0,
        reject=None,
        baseline=(None, -0.1),
        detrend=0,
        preload=True,
    )

    # Initialize list for aggregation
    bad_ch = []

    # Loop over sensor types
    for ch_type in [("eeg", "eeg"), ("meg", "mag"), ("meg", "grad")]:

        # Set switches for subsetting data for each modality
        if ch_type[0] == "eeg":
            eeg = True
            meg = False
            print("EEG")
        elif ch_type[0] == "meg":
            eeg = False
            meg = ch_type[1]
            print(meg.upper())

        # Subset data for chosen type
        picks = mne.pick_types(
            epo.info,
            meg=meg,
            eeg=eeg,
            stim=False,
            eog=False,
            include=[],
            exclude=[],
        )

        # Initialize Ransac object with defaults and fit to data
        ran = Ransac(verbose=True, picks=picks, n_jobs=-1, random_state=42)
        ran.fit(epo)

        # If bad channels found, append to aggregator list
        bads = ran.bad_chs_
        if bads:
            bad_ch += bads

    return bad_ch


def remove_bad_channels(inst, bad_ch):

    # Make a copy
    # inst = inst.copy()

    # Mark bad channels
    inst.info["bads"] = bad_ch

    # Interpolate, apply average ref to EEG and return
    inst.interpolate_bads(reset_bads=True, mode="accurate", origin="auto")
    inst = inst.set_eeg_reference("average")

    return inst
    