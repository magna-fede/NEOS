#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:06:22 2023

@author: fm02
"""


import mne

from mne.source_space import compute_distance_to_sensors
from mne.source_estimate import SourceEstimate
import matplotlib
import matplotlib.pyplot as plt

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

# matplotlib.use('Agg')
# mne.viz.set_3d_backend('pyvistaqt')

def run_make_forward_solution(sbj_id):

    subject = str(sbj_id)

    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])

    # doesn't matter which raw file, as long as transed
    raw_stem = config.sss_map_fnames[sbj_id][1][0]
    raw_fname = path.join(sbj_path, raw_stem + '.fif')

    src_fname = path.join(subjects_dir, str(sbj_id), 'bem', str(sbj_id) + '_' + str(config.src_spacing) + '-src.fif')

    print('Source space from: %s' % src_fname)
    src = mne.read_source_spaces(src_fname)

    # coordinate transformation
    trans_fname = path.join(sbj_path, f'{raw_stem}-trans.fif')

    fwd_fname = path.join(sbj_path, subject + '_EEGMEG-fwd_solved.fif')

    fwd = mne.read_forward_solution(fname=fwd_fname)
    
    mne.convert_forward_solution(fwd, surf_ori=True, copy=False)
    leadfield = fwd['sol']['data']
    print("Leadfield size : %d x %d" % leadfield.shape)

    grad_map = mne.sensitivity_map(fwd, ch_type='grad', mode='fixed')
    mag_map = mne.sensitivity_map(fwd, ch_type='mag', mode='fixed')
    eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')

    picks_meg = mne.pick_types(fwd['info'], meg=True, eeg=False)
    picks_eeg = mne.pick_types(fwd['info'], meg=False, eeg=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Lead field matrix (500 dipoles only)', fontsize=14)
    for ax, picks, ch_type in zip(axes, [picks_meg, picks_eeg], ['meg', 'eeg']):
        im = ax.imshow(leadfield[picks, :500], origin='lower', aspect='auto',
                       cmap='RdBu_r')
        ax.set_title(ch_type.upper())
        ax.set_xlabel('sources')
        ax.set_ylabel('sensors')
        fig.colorbar(im, ax=ax)
        
    fig.savefig(path.join(sbj_path, 'Figures', 'leadfield_solved.png'))
    plt.close('all')
    fig_2, ax = plt.subplots()
    ax.hist([grad_map.data.ravel(), mag_map.data.ravel(), eeg_map.data.ravel()],
            bins=20, label=['Gradiometers', 'Magnetometers', 'EEG'],
            color=['c', 'b', 'k'])
    fig_2.legend()
    ax.set(title='Normal orientation sensitivity',
           xlabel='sensitivity', ylabel='count')
    fig_2.savefig(path.join(sbj_path, 'Figures', 'Normal Orientation Sensitivity_solved.png'))
    
    brain_sens = grad_map.plot(
        subjects_dir=subjects_dir, clim=dict(lims=[0, 50, 100]), figure=1)
    brain_sens.add_text(0.1, 0.9, 'Gradiometer sensitivity', 'title', font_size=16)
    
    brain_sens.save_image(path.join(sbj_path, 'Figures', 'Gradiometers Sensitivity_solved.png'))
    brain_sens.close()    
    # source space with vertices
    src = fwd['src']
    
    # Compute minimum Euclidean distances between vertices and MEG sensors
    depths = compute_distance_to_sensors(src=src, info=fwd['info'],
                                         picks=picks_meg).min(axis=1)
    maxdep = depths.max()  # for scaling
    
    vertices = [src[0]['vertno'], src[1]['vertno']]
    
    depths_map = SourceEstimate(data=depths, vertices=vertices, tmin=0.,
                                tstep=1.)
    
    brain_dep = depths_map.plot(
        subject=subject, subjects_dir=subjects_dir,
        clim=dict(kind='value', lims=[0, maxdep / 2., maxdep]), figure=2)
    brain_dep.add_text(0.1, 0.9, 'Source depth (m)', 'title', font_size=16)
    brain_dep.save_image(path.join(sbj_path, 'Figures', 'Source depth_solved.png'))
        
    corr = np.corrcoef(depths, grad_map.data[:, 0])[0, 1]
    print('Correlation between source depth and gradiomter sensitivity values: %f.'
          % corr)
    brain_dep.close()
    plt.close('all')
    
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
            30
            ]

for sbj_id in sbj_ids:
    run_make_forward_solution(sbj_id)
    
