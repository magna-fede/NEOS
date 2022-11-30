#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:31:05 2022

@author: federica.magnabosco@mrc-cbu.cam.ac.uk
"""

import NEOS_config as config
import sys
import os
from os import path
import numpy as np
import pandas as pd

from importlib import reload
import pickle
import mne

import seaborn as sns
import matplotlib
matplotlib.use('Agg')  #  for running graphics on cluster ### EDIT

import matplotlib.pyplot as plt

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

reload(config)


reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

mne.viz.set_browser_backend("matplotlib")

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")

def evoked_sensors(sbj_id):
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    
    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]
    
    # only use files for correct conditions
    sss_map_fnames = []
    for sss_file in tmp_fnames:
        sss_map_fnames.append(sss_file)
    
    data_raw_file = path.join(sbj_path,
                               sbj_path[-3:] +
                              "_sss_f_ica_od_unfiltered_onraw_raw.fif")
    
    print(f'Reading raw file {sss_map_fnames}')
    data = mne.io.read_raw_fif(data_raw_file)
    
    devents = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_target_events.fif'))
    event_dict = {'FRP': 999}
    epochs = mne.Epochs(data, devents, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                    reject=reject_criteria, flat=flat_criteria,
                    preload=True)
    evoked = epochs['FRP'].average()
    
    FRP_fig = evoked.plot_joint(times=[0, .110, .167, .210, .266, .330, .430])

    for i, fig in zip(['EEG','MAG','GRAD'], FRP_fig):
        fname_fig = path.join(sbj_path, 'Figures', f'FRP_all_{i}_opticat_unfiltered_raw.png')
        fig.savefig(fname_fig)     
        
    rows = np.where(devents[:,2]==999)
    for row in rows[0]:
        if devents[row-2, 2] == 1:
            devents[row, 2] = 991
        elif devents[row-2, 2] == 2:
            devents[row, 2] = 992
        elif devents[row-2, 2] == 3:
            devents[row, 2] = 993
        elif devents[row-2, 2] == 4:
            devents[row, 2] = 994
        elif devents[row-2, 2] == 5:
            devents[row, 2] = 995
            
    event_dict = {'Abstract/Predictable': 991, 
                  'Concrete/Predictable': 992,
                  'Abstract/Unpredictable': 993, 
                  'Concrete/Unpredictable': 994}
    
    epochs = mne.Epochs(data, devents, picks=['meg', 'eeg', 'eog'], tmin=-0.3, tmax=0.7, event_id=event_dict,
                        reject=reject_criteria, flat=flat_criteria,
                        preload=True)
#    epochs = mne.read_epochs(path.join(sbj_path, sbj_path[-3:] +
#                      '_frp_percondition-epo.fif'))
    
    # evokeds_pred = [epochs[name].average() for name in ('Predictable', 'Unpredictable')]
    # evokeds_conc = [epochs[name].average() for name in ('Concrete', 'Abstract')]

    # predictability_fig = mne.viz.plot_evoked_topo(evokeds_pred)
    # concreteness_fig = mne.viz.plot_evoked_topo(evokeds_conc)

    # fname_fig = path.join(sbj_path, 'Figures', f'FRP_concreteness.png')
    # concreteness_fig.savefig(fname_fig)
    # fname_fig = path.join(sbj_path, 'Figures', f'FRP_predictability.png')
    # predictability_fig.savefig(fname_fig)      
    cond1 = 'Predictable'
    cond2 = 'Unpredictable'
    
    fig, ax = plt.subplots(3, 3)
    params = dict(spatial_colors=True, #show=False,
                  time_unit='s')
    epochs[cond1].average().plot(**params)
    fname_fig = path.join(sbj_path, 'Figures', 'FRP_predictable_opticat_unfiltered_raw.png')
    plt.savefig(fname_fig)
    epochs[cond2].average().plot(**params)
    fname_fig = path.join(sbj_path, 'Figures', 'FRP_unpredictable_opticat_unfiltered_raw.png')
    plt.savefig(fname_fig)
    contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()],
                                  weights=[1, -1])
    contrast.plot(**params)
    fname_fig = path.join(sbj_path, 'Figures', 'FRP_predictability_opticat_unfiltered_raw.png')
    plt.savefig(fname_fig)

    cond1 = 'Concrete'
    cond2 = 'Abstract'
    
    fig, ax = plt.subplots(3, 3)
    params = dict(spatial_colors=True, show=False,
                  time_unit='s')
    epochs[cond1].average().plot(**params)
    fname_fig = path.join(sbj_path, 'Figures', 'FRP_concrete_opticat_unfiltered_raw.png')
    plt.savefig(fname_fig)
    epochs[cond2].average().plot(**params)
    fname_fig = path.join(sbj_path, 'Figures', 'FRP_abstract_opticat_unfiltered_raw.png')
    plt.savefig(fname_fig)
    contrast = mne.combine_evoked([epochs[cond1].average(), epochs[cond2].average()],
                                  weights=[1, -1])
    contrast.plot(**params)
    fname_fig = path.join(sbj_path, 'Figures', 'FRP_concreteness_opticat_unfiltered_raw.png')
    plt.savefig(fname_fig)
    
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 18) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    evoked_sensors(ss)
        













