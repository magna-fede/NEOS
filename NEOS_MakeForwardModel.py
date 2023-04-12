#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:26:08 2023

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

    # coordinate transformation
    trans_fname = path.join(sbj_path, f'{raw_stem}-trans.fif')

    # # one-shell BEM for MEG
    # bem_fname = path.join(subjects_dir, subject, 'bem', subject + '_MEG-bem.fif')

    # print('BEM: %s' % bem_fname)
    # bem = mne.bem.read_bem_solution(bem_fname)

    # fwd_fname = path.join(sbj_path, subject + '_MEG-fwd.fif')
    # print('Making forward solution: %s.' % fwd_fname)

    # fwd_meg = mne.make_forward_solution(raw_fname, trans=trans_fname, src=src, bem=bem,
    #                                     meg=True, eeg=False, mindist=5.0, verbose=True)

    # mne.write_forward_solution(fname=fwd_fname, fwd=fwd_meg, overwrite=True)

    ### three-shell BEM for MEG
    bem_fname = path.join(subjects_dir, subject, 'bem', subject + '_EEGMEG-bem.fif')
    print('BEM: %s' % bem_fname)
    bem = mne.bem.read_bem_solution(bem_fname)

    fwd_fname = path.join(sbj_path, subject + '_EEGMEG-fwd.fif')
    print('Making forward solution: %s.' % fwd_fname)

    fwd_eegmeg = mne.make_forward_solution(info=raw_fname, trans=trans_fname, src=src, bem=bem,
                                            meg=True, eeg=True, mindist=5.0, 
                                            verbose=True)

    mne.write_forward_solution(fname=fwd_fname, fwd=fwd_eegmeg, overwrite=True)
    
    # ### three-shell BEM for EEG
    # bem_fname = path.join(subjects_dir, subject, 'bem', subject + '_EEGMEG-bem.fif')
    # print('BEM: %s' % bem_fname)
    # bem = mne.bem.read_bem_solution(bem_fname)

    # print('Making forward solution: %s.' % fwd_fname)

    # # fwd_eegmeg = mne.make_forward_solution(info=raw_fname, trans=trans_fname, src=src, bem=bem,
    # #                                         meg=True, eeg=True, mindist=5.0, 
    # #                                         verbose=True)
    
    # fwd_eeg = mne.make_forward_solution(info=raw_fname, trans=trans_fname, src=src, bem=bem,
    #                                         meg=False, eeg=True, mindist=5.0, 
    #                                         verbose=True)
    
    # fwd_meg = mne.make_forward_solution(info=raw_fname, trans=trans_fname, src=src, bem=bem,
    #                                         meg=True, eeg=False, mindist=5.0, 
    #                                         verbose=True)
    
    # from mne.forward import _merge_fwds
    # fwd = _merge_fwds(dict(meg=fwd_meg, eeg=fwd_eeg))
    
    # mislab_ch = fwd_eegmeg['sol']['row_names']
    # correct_ch = fwd_eegmeg['info']['ch_names']
    
    # fwd_eegmeg['sol']['data'] = pd.DataFrame(fwd_eegmeg['sol']['data'], index=mislab_ch).loc[correct_ch, :].values
    # fwd_eegmeg['sol']['row_names'] = correct_ch    

    # mne.write_forward_solution(fname=fwd_fname, fwd=fwd_eeg, overwrite=True)
# # get all input arguments except first
# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]

# for ss in sbj_ids:

#     run_make_forward_solution(ss)

# print('Done.')