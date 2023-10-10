#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:35:29 2023

@author: fm02
"""

from scipy import stats as stats

import pandas as pd
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sys
import os
from os import path
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mpl_toolkits.axes_grid1 import make_axes_locatable

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

stcs_path = path.join(config.data_path, "stcs")

sbj_ids = [
            1,
            2,
            3,
        #   4, #fell asleep
            5,
            6,
        #    7, #no MRI
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18, 
            19, 
        #   20, #too magnetic to test
            21,
            22, 
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30
            ]

src_fname = path.join(config.subjects_dir, "fsaverage", "bem", "fsaverage-ico-5-src.fif")

src = mne.read_source_spaces(src_fname)
vertices = [src[0]['vertno'], src[1]['vertno']]

predictables = [mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Predictable_eLORETA_MEGonly_auto_dropbads_fsaverage")) \
                if sbj_id==12 else \
                    mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Predictable_eLORETA_EEGMEGauto_dropbads_fsaverage")) \
                        for sbj_id in sbj_ids ]

p = [np.array(stc.data) for stc in predictables]
avg_p = np.stack(p).mean(axis=0)
GA_unfold = mne.SourceEstimate(avg_p, vertices=vertices, tmin=-0.152, tstep=0.004)

cropped_screenshot = list()
for i in np.arange(-1, 6, 0.5):
    clim = dict(kind="value", lims=[1.2e-11, 1.248e-11, 1.5e-11])
    brain = GA_unfold.plot(
        views="lat",
        hemi="split",
        surface="pial_semi_inflated",
        size=(800, 400),
        subject="fsaverage",
        subjects_dir=config.subjects_dir,
        initial_time=0.1*i,
        background="w",
        time_viewer=False,
        show_traces=False,
        clim=clim,
        colorbar=None,
        cortex='low_contrast',
    )
    screenshot = brain.screenshot()
    brain.close()
    
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot.append(screenshot[nonwhite_row][:, nonwhite_col])

fig, axs = plt.subplots(nrows=len(cropped_screenshot)+1, ncols=1,
                        sharex=True, sharey=True, figsize=(8,20))

for f, ax in zip(cropped_screenshot, axs.ravel()):
    ax.imshow(f)
    ax.axis('off')
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("right", size="5%")
cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap='hot', label="Activation (F)")
#fig.colorbar(cbar)
plt.savefig('GA_unfold_predictable.png', bbox_inches='tight')


unpredictables = [mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Unpredictable_eLORETA_MEGonly_auto_dropbads_fsaverage")) \
                if sbj_id==12 else \
                    mne.read_source_estimate(os.path.join(stcs_path, f"{sbj_id}_unfold_stc_Unpredictable_eLORETA_EEGMEGauto_dropbads_fsaverage")) \
                        for sbj_id in sbj_ids ]

unp = [np.array(stc.data) for stc in unpredictables]
avg_unp = np.stack(unp).mean(axis=0)
GA_unpunfold = mne.SourceEstimate(avg_unp, vertices=vertices, tmin=-0.152, tstep=0.004)


cropped_screenshot = list()
for i in np.arange(-1, 6, 0.5):
    clim = dict(kind="value", lims=[1.2e-11, 1.248e-11, 1.5e-11])
    brain = GA_unpunfold.plot(
        views="lat",
        hemi="split",
        surface="pial_semi_inflated",
        size=(800, 400),
        subject="fsaverage",
        subjects_dir=config.subjects_dir,
        initial_time=0.1*i,
        background="w",
        time_viewer=False,
        show_traces=False,
        clim=clim,
        colorbar=None,
        cortex='low_contrast',
    )
    screenshot = brain.screenshot()
    brain.close()
    
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot.append(screenshot[nonwhite_row][:, nonwhite_col])

fig, axs = plt.subplots(nrows=len(cropped_screenshot)+1, ncols=1,
                        sharex=True, sharey=True, figsize=(8,20))

for f, ax in zip(cropped_screenshot, axs.ravel()):
    ax.imshow(f)
    ax.axis('off')
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("right", size="5%")
cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap='hot', label="Activation (F)")
#fig.colorbar(cbar)
plt.savefig('GA_unfold_unpredictable.png', bbox_inches='tight')





X = np.stack(p) - np.stack(unp)
out = stats.ttest_1samp(X, 0, axis=0)

t_values = out[0]
for i in range(len(t_values)):
    t_values[i][abs(t_values[i])<2] = 0
    
t_maps = mne.SourceEstimate(t_values, vertices=vertices, tmin=-0.152, tstep=0.004)
cropped_screenshot = list()
for i in np.arange(-1, 6, 0.5):
    clim = dict(kind="value", lims=[-3, 0, 3])
    brain = t_maps.plot(
        views="lat",
        hemi="split",
        surface="pial_semi_inflated",
        size=(800, 400),
        subject="fsaverage",
        subjects_dir=config.subjects_dir,
        initial_time=0.1*i,
        background="w",
        time_viewer=False,
        show_traces=False,
        clim=clim,
        colorbar=None,
        colormap='mne',
        cortex='low_contrast',
    )
    screenshot = brain.screenshot()
    brain.close()
    
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot.append(screenshot[nonwhite_row][:, nonwhite_col])

fig, axs = plt.subplots(nrows=len(cropped_screenshot)+1, ncols=1,
                        sharex=True, sharey=True, figsize=(8,20))

for f, ax in zip(cropped_screenshot, axs.ravel()):
    ax.imshow(f)
    ax.axis('off')
divider = make_axes_locatable(axs[-1])
cax = divider.append_axes("right", size="5%")
cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap='mne', label="T-values uncorrected")
#fig.colorbar(cbar)
plt.savefig('GA_unfold_tmap.png', bbox_inches='tight')
