#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics to compute for evaluating SNR from fixation and saccade locked epochs.
Currently estimates:
    From fixation-locked epochs (in function fixation_locked):
        1. GFP of first 100ms of baseline
        2. GFP of last 100ms
        3. Ratio of the above two expressed as GFP of last over first 100ms
        4. Peak amplitude of P1 at sensor EEG008
        5. Peak latency of P1 at sensor EEG008
    From saccade-locked epochs (in function saccade_locked):
        1. Peak amplitude of Saccade at sensor EEG001
        2. Peak latency of Saccade at sensor EEG001
        3. Area under the curve of Saccade at sensor EEG001 in first 40msec
Optional plotting arguments produce plots with detected peaks for P1 & Saccade.
@author: py01
"""

import mne
from scipy.integrate import simpson
from matplotlib import pyplot as plt


def fixation_locked(raw, evt, plot=False):
    """
    Compute P1 peak from fixation-locked epochs at posterior sensor EEG008, and
    GFP in first 100ms of baseline and last 100ms of epoch.
    Parameters
    ----------
    raw : mne.io.fiff.raw.Raw
        MNE Raw object.
    evt : np.ndarray
        Event structure in MNE-format.
    plot : bool, optional
        To plot or not to plot. The default is False.
    Returns
    -------
    dict or (dict, fig)
        Returns dict with estimates or a tuple of dict with estimates and a
        figure object with plot.
    """
    # Create evoked object with 901 (start Fixation) as event
    evo = mne.Epochs(
        raw,
        evt,
        picks=["eeg"],
        event_id=901,
        tmin=-0.2,
        tmax=1,
        baseline=(None, 0),
    ).average()

    # Get P1 amplitude
    chan, lat, peak = (
        evo.copy()
        .pick_channels(["EEG001"])
        .get_peak(
            tmin=0.075,
            tmax=0.125,
            mode="pos",
            return_amplitude=True,
        )
    )

    # Get GFP
    gfp_baseline = evo.get_data(tmin=-0.2, tmax=-0.1).std(axis=0).mean()
    gfp_n400 = evo.get_data(tmin=0.25, tmax=0.4).std(axis=0).mean()
    gfp_last100 = evo.get_data(tmin=0.9, tmax=1).std(axis=0).mean()
    gfp_ratio = gfp_last100 / gfp_baseline

    # Return dict of outputs
    out = {
        "P1_amplitude": peak,
        "P1_latency": lat,
        "GFP_first100": gfp_baseline,
        "GFP_n400": gfp_n400,
        "GFP_last100": gfp_last100,
        "GFP_ratio": gfp_ratio,
    }

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        x = evo.times
        y = evo.get_data(picks=chan).squeeze()

        ax.plot(x, y, color="dodgerblue")
        ax.plot(
            lat,
            peak,
            color="indianred",
            ms=5,
            marker="o",
            label="Peak of P1",
            ls=None,
            lw=0,
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title("Fixation-locked activity at EEG001")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(frameon=False, title="")

        fig.show()

        return out, fig

    return out


def saccade_locked(raw, evt, plot=False):
    """
    Compute saccade peak from saccade-locked epochs at frontal sensor EEG001,
    and estimate area-under-the-curve in first 40ms of saccade.
    Parameters
    ----------
    raw : mne.io.fiff.raw.Raw
        MNE Raw object.
    evt : np.ndarray
        Event structure in MNE-format.
    plot : bool, optional
        To plot or not to plot. The default is False.
    Returns
    -------
    dict or (dict, fig)
        Returns dict with estimates or a tuple of dict with estimates and a
        figure object with plot.
    """
    # Create evoked object with 801 (start Saccade) as event
    evo = mne.Epochs(
        raw,
        evt,
        picks=["eeg"],
        event_id=801,
        tmin=-0.02,
        tmax=0.05,
        baseline=(None, 0),
    ).average()

    # Get Area-under-curve for saccade
    tmin = 0
    tmax = 0.04
    times = evo.times[(evo.times >= tmin) & (evo.times < tmax)]
    amplitude = evo.get_data(picks="EEG008", tmin=tmin, tmax=tmax).squeeze()
    auc = simpson(y=abs(amplitude), x=times, dx=1 / evo.info["sfreq"])

    # Get peak amplitude of saccade
    chan, lat, peak = (
        evo.copy()
        .pick_channels(["EEG008"])
        .get_peak(
            tmin=0.0,
            tmax=0.04,
            mode="pos",
            return_amplitude=True,
        )
    )

    # Return dict of outputs
    out = {
        "S_amplitude": peak,
        "S_latency": lat,
        "S_auc": auc,
    }

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        x = evo.times
        y = evo.get_data(picks=chan).squeeze()

        ax.plot(x, y, color="dodgerblue")
        ax.plot(
            lat,
            peak,
            color="indianred",
            ms=5,
            marker="o",
            label="Peak of Saccade",
            ls=None,
            lw=0,
        )
        ax.fill_between(
            x=times, y1=amplitude, alpha=0.25, color="dodgerblue", label="AUC"
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title("Saccade-locked activity at EEG008")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(frameon=False, title="")

        fig.show()

        return out, fig

    return out


def compute_metrics(raw, evt, plot=False):
    """
    Compute both fixation and saccade locked metrics. Wrapper around the
    functions fixation_locked and saccade_locked. See their respective docs for
    details.
    Parameters
    ----------
    raw : mne.io.fiff.raw.Raw
        MNE Raw object.
    evt : np.ndarray
        Event structure in MNE-format.
    plot : bool, optional
        To plot or not to plot. The default is False.
    Returns
    -------
    dict or (dict, fig)
        Returns dict with estimates or a tuple of dict with estimates and list
        of figure objects.
    """

    # If plotting requested, return both estimates and figures
    if plot:
        out1, fig1 = fixation_locked(raw, evt, plot)
        out2, fig2 = saccade_locked(raw, evt, plot)
        out1.update(out2)
        return out1, [fig1, fig2]

    # Else return only estimates
    else:
        out1 = fixation_locked(raw, evt)
        out2 = saccade_locked(raw, evt)
        out1.update(out2)
        return out1