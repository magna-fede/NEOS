#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 18:15:48 2023

@author: fm02
"""

import numpy as np
import matplotlib.pyplot as plt

import mne

from mne.datasets import sample

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
import pickle

import matplotlib.pyplot as plt
import seaborn as sns


stc_path = path.join(config.data_path, "stcs")
subjects_dir = config.subjects_dir
labels_dir = path.join(config.data_path, "my_ROIs")
labels_path = path.join(config.data_path, "my_ROIs")


fname_fsaverage_src = path.join(subjects_dir,
                                'fsaverage',
                                'bem', 
                                'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(fname_fsaverage_src)


stc = mne.read_source_estimate(path.join(stc_path, '1_stc_predictable_normalorientation_fsaverage'))

lATL = mne.read_label(path.join(labels_path, 'l_ATL-lh.label'))
rATL = mne.read_label(path.join(labels_path, 'r_ATL-rh.label'))
PVA = mne.read_label(path.join(labels_path, 'PVA-lh.label'))
IFG = mne.read_label(path.join(labels_path, 'IFG-lh.label'))
AG = mne.read_label(path.join(labels_path, 'AG-lh.label'))
PTC = mne.read_label(path.join(labels_path, 'PTC-lh.label'))

times=np.arange(-300,701,1)

rois = [lATL,
        rATL, 
        PVA,
        IFG,
        AG,
        PTC]

stc_rois = dict()
stc_rois = dict()

for roi in rois:
    stc_rois[roi] = stc.in_label(roi)

stc_rois = dict()
avgs = dict()

for roi in rois:
    stc_rois[roi] = stc.in_label(roi)
    avgs[roi] = stc.extract_label_time_course(roi, src, mode='mean_flip')
    
    fig, ax = plt.subplots(1);
    ax.plot(times, stc_rois[roi].data.T, 'k', linewidth=0.5, alpha=0.5);
    ax.plot(times, avgs[roi].T, linewidth=2)
    ax.set(xlabel='Time (ms)', ylabel='Source amplitude',
       title='Activations in Label %r' % (roi.name))
    plt.show()
    


