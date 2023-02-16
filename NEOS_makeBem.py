#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create BEM model.
Adapted from FPVS https://github.com/olafhauk/FPVS_sweep

@author: fm02
"""

import sys

import os
from os import path

import numpy as np

import importlib
from importlib import reload

import mne

import NEOS_config as config
reload(config)

# Whether to create or update symbolic links or not
# Note: This would overwrite any links manually created,
# e.g. after shrinking inner skull
do_links = True

# set Freesurfer environment variables
filename = "/imaging/local/software/mne_python/set_MNE_2.7.3_FS_6.0.0_environ.py"
# for Python 3 instead of execfile
exec(compile(open(filename, "rb").read(), filename, 'exec'))

# where MRIs are
subjects_dir = config.subjects_dir

# default conductivities
conductivity_1 = (0.3,)  # for single layer
conductivity_3 = (0.3, 0.006, 0.3)  # for three layers


def run_make_bem(sbj_id):

    subject = sbj_id

    if subject == '':

        print('No subject name for MRI specified - doing nothing now.')

        return

    if do_links:

        print('Updating links to surfaces.')
        cmd_list = []
        sd = config.subjects_dir
        sb = str(subject)

        cmd_list.append('ln -sf %s/%s/bem/watershed/%s_inner_skull_surface \
                        %s/%s/bem/inner_skull.surf' % (sd, sb, sb, sd, sb))
        cmd_list.append('ln -sf %s/%s/bem/watershed/%s_outer_skull_surface \
                        %s/%s/bem/outer_skull.surf' % (sd, sb, sb, sd, sb))
        cmd_list.append('ln -sf %s/%s/bem/watershed/%s_outer_skin_surface \
                        %s/%s/bem/outer_skin.surf' % (sd, sb, sb, sd, sb))
        cmd_list.append('ln -sf %s/%s/bem/watershed/%s_brain_surface \
                        %s/%s/bem/brain_surface.surf' % (sd, sb, sb, sd, sb))

        [os.system(cmd) for cmd in cmd_list]
        
        # note that for participants for which shrink_inner_skull was ran,
        # the corrected surfaces are in the watershed3 folder:
        
        # print('Updating links to surfaces.')
        # cmd_list = []
        # sd = config.subjects_dir
        # sb = str(subject)

        # cmd_list.append('ln -sf %s/%s/bem/watershed3/%s_inner_skull_surface \
        #                 %s/%s/bem/inner_skull.surf' % (sd, sb, sb, sd, sb))
        # cmd_list.append('ln -sf %s/%s/bem/watershed3/%s_outer_skull_surface \
        #                 %s/%s/bem/outer_skull.surf' % (sd, sb, sb, sd, sb))
        # cmd_list.append('ln -sf %s/%s/bem/watershed3/%s_outer_skin_surface \
        #                 %s/%s/bem/outer_skin.surf' % (sd, sb, sb, sd, sb))
        # cmd_list.append('ln -sf %s/%s/bem/watershed3/%s_brain_surface \
        #                 %s/%s/bem/brain_surface.surf' % (sd, sb, sb, sd, sb))

        # [os.system(cmd) for cmd in cmd_list]

    else:

        print('###\nNOT updating symbolic links to surfaces.\n###')

    print('Making BEM model for %s.' % subject)

    # print('Creating BEM surfaces.')
    mne.bem.make_watershed_bem(subject=str(subject),
                                subjects_dir=config.subjects_dir,
                                overwrite=True)
    # one-shell BEM for MEG
    print('###\nMEG\n###')

    model = mne.make_bem_model(subject=str(subject), ico=4,
                               conductivity=conductivity_1,
                               subjects_dir=subjects_dir)

    bem = mne.make_bem_solution(model)

    bem_fname = path.join(subjects_dir, str(subject), 'bem', str(subject) + '_MEG-bem.fif')

    print('Writing BEM solution for MEG to file %s.' % bem_fname)
    mne.bem.write_bem_solution(bem_fname, bem, overwrite=True)

    # three-shell BEM for EEG+MEG
    print('EEG+MEG')
    model = mne.make_bem_model(subject=str(subject), ico=4,
                               conductivity=conductivity_3,
                               subjects_dir=subjects_dir)

    bem = mne.make_bem_solution(model)

    bem_fname = path.join(subjects_dir, str(subject), 'bem', str(subject) + '_EEGMEG-bem.fif')

    print('Writing BEM solution for EEG+MEG to file %s.' % bem_fname)

    mne.bem.write_bem_solution(bem_fname, bem, overwrite=True)

sbj_ids = [
            1, # run watershed
            2,
            3, # run watershed
        #   4,
            5,
            6,
        #   7,
            8,
            9, # run watershed
            10,
            11,
            12,
            13,
            14,
            15, # run watershed #BIG PROBLEM => occipital skull not aligned
            16,
            17, # corrected T1 for some extra points (probably the subject moved)
            18, # run watershed
            19, 
        #   20,
            21,
            22, 
            23, 
            24,
            26,
            27
            ]

for ss in sbj_ids:

    run_make_bem(ss)

print('Done.')
