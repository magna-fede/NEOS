#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:01:51 2023

@author: fm02
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import stats as stats

import pandas as pd
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sys
import os

import mne
from mne.stats import permutation_cluster_1samp_test
from mne.channels import find_ch_adjacency
from mne.viz import plot_compare_evokeds

from os import path

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
sbj_ids = [
            1,
            2,
            3,
        #   4, #fell asleep
            5,
            6,
        #    7, #no MRI
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
        #   20, #too magnetic to test
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

data_path = path.join(config.data_path, "AVE")
cluster_stats = dict.fromkeys(['grad', 'mag', 'eeg'])
for key in cluster_stats.keys():
    cluster_stats[key] = dict.fromkeys(['Concreteness', 'Predictability'])

for ch_type in ['grad', 'mag', 'eeg']:
    evokeds = dict.fromkeys(['Abstract', 'Concrete', 'Predictable', 'Unpredictable'])

    for key in evokeds.keys():
        evokeds[key] = list()
    
    for condition in evokeds.keys():
        for sbj_id in sbj_ids:
            evoked = mne.read_evokeds(path.join(data_path, f"{sbj_id}_{condition}_unfold_evoked-ave.fif"))[0]
            if (sbj_id==12) & (ch_type=='eeg'):
                pass
            else:                
                evokeds[condition].append(evoked.get_data(picks=ch_type))
    
    for test, c_1, c_2 in zip(['Concreteness', 'Predictability'],
                              ['Abstract', 'Unpredictable'],
                              ['Concrete', 'Predictable']):
        c_1 = np.stack(evokeds[c_1])
        c_2 = np.stack(evokeds[c_2])
        
        X = c_1 - c_2
        X = np.transpose(X, (0, 2, 1)) 
        
        adjacency, ch_names = find_ch_adjacency(evoked.info, ch_type=ch_type)
        
        p_threshold = 0.001
        df = len(X) - 1  # degrees of freedom for the test
        t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
        
        # run the cluster based permutation analysis
        cluster_stats[ch_type][test] = permutation_cluster_1samp_test(
            X,
            n_permutations=10000,
            threshold=t_threshold,
            n_jobs=-1,
            adjacency=adjacency,
        )

times = np.arange(-0.152, 0.500, 0.004)

for ch_type in cluster_stats.keys():
    for test in cluster_stats[ch_type]:
        if any(cluster_stats[ch_type][test][2] <0.01):
            print(f"{test} {ch_type} has significant clusters")
            
preds = mne.read_evokeds(path.join(data_path, 'GA_unfold_predictable-ave.fif'))[0]
unpreds = mne.read_evokeds(path.join(data_path, 'GA_unfold_unpredictable-ave.fif'))[0]

test = 'Predictability'


for ch_type in cluster_stats.keys():
    evo_p = preds.copy().pick(ch_type)
    evo_u = unpreds.copy().pick(ch_type)
    
    evo_p.comment = 'Predictable'
    evo_u.comment = 'Unpredictable'

    evokeds = [evo_p, evo_u]
    diff_wave = mne.combine_evoked([evo_u, evo_p], [1, -1])
    
    # We subselect clusters that we consider significant at an arbitrarily
    # picked alpha level: "p_accept".
    # NOTE: remember the caveats with respect to "significant" clusters that
    # we mentioned in the introduction of this tutorial!
    p_accept = 0.01
    good_cluster_inds = np.where(cluster_stats[ch_type][test][2] < p_accept)[0]
    
    # configure variables for visualization
    colors = sns.color_palette(['#FFBE0B',
                                '#FF006E',
                                ])
    
    # organize data for plotting

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(cluster_stats[ch_type][test][1][clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
    
        # get topography for F stat
        f_map = cluster_stats[ch_type][test][0][time_inds, ...].mean(axis=0)
    
        # get signals at the sensors contributing to the cluster
        sig_times = times[time_inds]
    
        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
    
        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
    
        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], evokeds[0].info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            cmap="Reds",
            vlim=(np.min, np.max),
            show=False,
            colorbar=False,
            scalings=dict(eeg=1, grad=1, mag=1),
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]
    
        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")
    
        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)
    
        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )
    
        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        # title = f"Cluster #{i_clu + 1} {ch_type}, {len(ch_inds)} sensor"
        # if len(ch_inds) > 1:
        #     title += "s (mean)"
        title = f"Cluster #{i_clu + 1} {ch_type}, Difference across {len(ch_inds)} sensors"

        # plot_compare_evokeds(
        #     evokeds,
        #     title=title,
        #     picks=ch_inds,
        #     axes=ax_signals,
        #     colors=colors,
        #     show=False,
        #     split_legend=True,
        #     truncate_yaxis="auto",
        # )
        plot_compare_evokeds(
            diff_wave,
            title=title,
            picks=ch_inds,
            axes=ax_signals,
            colors=colors,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )
        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )
    
        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=0.05)
        fig.savefig(f"/home/fm02/MEG_NEOS/plots/sensor_unfold_{i_clu + 1}_{ch_type}_diff.png")

        plt.show()
        
