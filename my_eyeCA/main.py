#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:58:21 2022

@author: py01
"""

from pathlib import Path
import mne
import os

os.chdir("/home/py01/Projects/EyeCA/src")
import preprocess
import ica

# %%
file = Path("/home/py01/Projects/EyeCA/data/block1_sss_f_raw.fif")
raw = mne.io.read_raw(file)

file = Path("/home/py01/Projects/EyeCA/data/165_all_events.fif")
evt = mne.read_events(file)

# %%
evt = preprocess.subset_events(raw, evt)

# %%
draw, devt = preprocess.downsample_and_filter(raw, evt, lf=2, hf=40, sf=False)

# %%
# bad_ch = preprocess.find_bad_channels(draw, devt)
# draw = preprocess.remove_bad_channels(draw, bad_ch)

# %%
cov = ica.compute_noise_covariance(draw)

# Visualize
# fig_cov, fig_svd = cov.plot(draw.info, show=True)

# %%
ic = ica.compute_ica(
    draw, cov, picks=["eeg", "meg"], method="extinfomax", n_comp=50
)
ic.save(
    "/home/py01/Projects/EyeCA/data/block1_sss_f_raw-ica.fif", overwrite=True
)

# %%
ic = mne.preprocessing.read_ica(
    "/home/py01/Projects/EyeCA/data/block1_sss_f_raw-ica.fif"
)
