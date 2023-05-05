#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:40:45 2023

@author: fm02
"""

import numpy as np

import mne

import sys
import os
from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config


ave_path = path.join(config.data_path, "AVE")
stc_path = path.join(config.data_path, "stcs")
method = "MNE"
snr = 3.
lambda2 = 1. / snr ** 2

conditions = ['predictable', 'unpredictable', 'abstract', 'concrete']
# conditions = ['abspred', 'absunpred', 'concpred', 'concunpred']

subjects_dir = config.subjects_dir

def compute_morphed_stcs(sbj_id):

    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])    
    subject = str(sbj_id)
    
    fname_src = path.join(subjects_dir,
                          subject,
                          'bem',
                          subject + '_' + str(config.src_spacing)
                          + '-src.fif')
    
    fname_fwd = path.join(sbj_path, 
                          subject + '_EEG-fwd_solved.fif')
    fname_fsaverage_src = path.join(subjects_dir,
                                    'fsaverage',
                                    'bem', 
                                    'fsaverage-ico-5-src.fif')
    fname_stc = path.join(stc_path,
                          f"{subject}_stc_EEGonly_predictable")
    
    stc = mne.read_source_estimate(fname_stc, subject=subject)
    
    src_orig = mne.read_source_spaces(fname_src)
    print(src_orig)
    
    fwd = mne.read_forward_solution(fname_fwd)
    print(fwd['src'])  
    
    print([len(v) for v in stc.vertices])
    
    src_to = mne.read_source_spaces(fname_fsaverage_src)
    print(src_to[0]['vertno'])  # special, np.arange(10242)
    morph = mne.compute_source_morph(stc, subject_from=subject,
                                     subject_to='fsaverage', src_to=src_to,
                                     subjects_dir=subjects_dir)
    for condition in conditions:
        print(f'Reading stc file for {condition} condition')
        fname_stc = path.join(stc_path,
                              f"{subject}_stc_EEGonly_{condition}")
                       
        print(fname_stc)
        stc = mne.read_source_estimate(fname_stc, subject=subject)
        stc_mph = morph.apply(stc)
        print(f'Saving morphed stc file for {condition}')
        fname_mph = path.join(stc_path, 
                              f"{subject}_stc_EEGonly_{condition}_fsaverage")
        print(fname_mph)            
        stc_mph.save(fname_mph)

if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
               21,22,23,24,25,26,27,28,29,30]


else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    compute_morphed_stcs(ss) 
    