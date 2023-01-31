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

import seaborn as sns

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

os.chdir("/home/fm02/MEG_NEOS/NEOS/my_eyeCA")
from my_eyeCA import apply_ica

os.chdir("/home/fm02/MEG_NEOS/NEOS")

mne.viz.set_browser_backend("matplotlib")

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")
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

bad_eeg = config.bad_channels[sbj_id]['eeg']

evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                      '_target_events.fif'))
    

evts_blocks = evts.copy()
data_raw_files = []
for raw_stem_in in sss_map_fnames:
    data_raw_files.append(
        path.join(path.join(sbj_path, raw_stem_in[:6] +
                            "_sss_f_ica_ovrw_both_raw.fif")))
   
raw_block = []
	# %%
for block, drf in enumerate(data_raw_files):
    raw = mne.io.read_raw(drf, preload=True)
    raw.info['bads'] = bad_eeg
    raw_block.append(raw)
    
data = mne.concatenate_raws(raw_block)

data_orgi = data.copy()

apply_ica.plot_evoked_sensors(data=data, devents=evts_blocks, comp_sel='filt0_1Hz')

# %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
data.filter(l_freq=0.5, h_freq=None)
evts_blocks = evts.copy()
apply_ica.plot_evoked_sensors(data=data, devents=evts_blocks, comp_sel='filt0_5Hz')

# %% High-pass raw data at 1Hz & compute SNR metrics on ICA-reconstructed data
data.filter(l_freq=1, h_freq=None)
evts_blocks = evts.copy()
apply_ica.plot_evoked_sensors(data=data, devents=evts_blocks, comp_sel='filt1_0Hz')

 
    
  
    
  
    
  
    
  
    
  
    
  
    
  