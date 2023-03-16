#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:06:22 2023

@author: fm02
"""


import mne

import sys
import os
from os import path

import numpy as np
import pandas as pd

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

subjects_dir = config.subjects_dir

def run_make_forward_solution(sbj_id):

    subject = str(sbj_id)

    print('Making Forward Solution for %s.' % subject)

    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    # doesn't matter which raw file, as long as transed
    raw_stem = config.sss_map_fnames[sbj_id][1][0]
    raw_fname = path.join(sbj_path, raw_stem + '.fif')

    src_fname = path.join(subjects_dir, str(sbj_id), 'bem', str(sbj_id) + '_' + str(config.src_spacing) + '-src.fif')

    print('Source space from: %s' % src_fname)
    src = mne.read_source_spaces(src_fname)
    
    plot_bem_kwargs = dict(
        subject=subject, subjects_dir=subjects_dir,
        brain_surfaces='white', orientation='coronal',
        slices=[50, 100, 150, 200])
    mne.viz.plot_bem(src=src, **plot_bem_kwargs)
    
sbj_ids = [ 1,
            2,
            3,
            5,
            6,
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
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            # 30
            ]

for sbj_id in sbj_ids:
    run_make_forward_solution(sbj_id)
    
