#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:27:40 2023

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

vertices = [src[0]['vertno'], src[1]['vertno']]

stcs_movies = []
sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]
for sbj_id in sbj_ids:
    stc_movie = mne.read_source_estimate(path.join(stc_path, f'{sbj_id}_stc_predictable_fsaverage'))
    
    stcs_movies.append(stc_movie.data)

GA_predictable = np.stack(stcs_movies).mean(axis=0)

GA_predictable = mne.SourceEstimate(GA_predictable, vertices=vertices, tmin=-0.3, tstep=0.001)

brain = GA_predictable.plot(
    subject='fsaverage', subjects_dir=config.subjects_dir, 
    hemi='both', views='lateral')

brain.save_movie(filename='ENG_met.mp4',tmin=0.0, interpolation='linear',\
                 time_dilation=4, framerate=10)
brain.close()




