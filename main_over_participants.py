#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:58:21 2022

@author: fm02
adapted from py01 EyeCA
"""

import sys
import os
from os import path


import mne
from mne.preprocessing import ICA, create_eog_epochs
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import preprocess
from my_eyeCA import ica


os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

mne.viz.set_browser_backend("matplotlib")

# %%
# Set MNE's log level to DEBUG
mne.set_log_level(verbose="DEBUG")

sbj_id = int(sys.argv[1])

sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

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

bad_eeg = config.bad_channels[sbj_id]['eeg']


	# %%
for drf in data_raw_files:
    raw = mne.io.read_raw(drf)
    raw.info['bads'] = bad_eeg

    evt_file = path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              '_all_events.fif')   
    evt = mne.read_events(evt_file)
    
	# %%
    ic = ica.run_ica_pipeline(
	    raw=raw, evt=evt, method="extinfomax", cov_estimator=None, n_comp=0.99
	)