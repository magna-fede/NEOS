#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a HTML report where 
    - for each participant
        - for each condition
            we plot
                comparison of overweighting vs no overweighting
                comparison of different ICA components selection
                comparison of high-pass filtering
                SNR aggreagates for all the above

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
import matplotlib.image as mpimg

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

reload(config)

report = mne.Report(title='Systematic comparison of ICA pipelines')

subjs = [
            1,
            2,
            3,
        #   4,
            5,
            6,
        #    7,
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
        #   20,
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

end = ".png"


ch_type = {
     'eeg': 'FRP_all_EEG_',                    
     'grad': 'FRP_all_GRAD_',
     'mag': 'FRP_all_MAG_',   
     }     

ch_type01 = {
     'eeg': 'FRP_EEG_all_',                    
     'grad': 'FRP_GRAD_all_',
     'mag': 'FRP_MAG_all_',   
     }     
        
conditions = {#"preica" : "_pre-ICA_",
             "eog" : "eog_",
             "var" : "var_",
             "both" : "both_"
             }

over = {"overweighted" : "ovrw_",
        "non-overweighted" : "",
        "onset overweighted" : "ovrwonset_"
        }

# uncorrected = {
#      'eeg': 'uncorrected_fixation_EEG.png',
#      'grad': 'uncorrected_fixation_GRAD.png',
#      'mag': 'uncorrected_fixation_MAG.png',
#     # 'uncorrected_saccade_EEG.png',
#     # 'uncorrected_saccade_GRAD.png',
#     # 'uncorrected_saccade_MAG.png',
#     }

# radar_plots = {
#     'filtering' : 'snr_EEG_filtering.png',
#     'component_selection' : 'snr_EEG_all_01Hz.png'
#     }

sbjs = [1,2]
for sbj_id in subjs:
    print(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0], 'Figures')

    # figs = []
    # captions= []    
    # for unc in uncorrected.values():
    #     section = f'Subject {sbj_id} - uncorrected'

    #     image_path = sbj_path + '/' + unc
            
    #     try:
    #         fig, ax = plt.subplots()
    #         img = mpimg.imread(image_path)
    #         ax.imshow(img)
    #         ax.set_axis_off()

    #         figs.append(fig)
    #         captions.append(unc)
    #     except:
    #         image_path = image_path[:-3]+'jpg'
    #         fig, ax = plt.subplots()
    #         img = mpimg.imread(image_path)
    #         ax.imshow(img)
    #         ax.set_axis_off()

    #         figs.append(fig)
    #         captions.append(unc)
        
    # report.add_figure(
    #     fig=figs, title='Uncorrected', section = section,
    #     caption=captions
    #     )
    # plt.close('all')
    
    figs = []
    captions= []    
    # for rplot in radar_plots.values():
    #     section = f'Subject {sbj_id} - summary'

    #     image_path = sbj_path + '/' + rplot
            
    #     try:
    #         fig, ax = plt.subplots()
    #         img = mpimg.imread(image_path)
    #         ax.imshow(img)
    #         ax.set_axis_off()

    #         figs.append(fig)
    #         captions.append(rplot)
    #     except:
    #         image_path = image_path[:-3]+'jpg'
    #         fig, ax = plt.subplots()
    #         img = mpimg.imread(image_path)
    #         ax.imshow(img)
    #         ax.set_axis_off()

    #         figs.append(fig)
    #         captions.append(rplot)
        
    # report.add_figure(
    #     fig=figs, title='Summary', section = section,
    #     caption=captions
    #     )
    # plt.close('all')    

    for ch in ch_type.keys():
        for ovr in over.keys():
            section = f'Subject {sbj_id} - ICA {ch} {ovr}'
            figs = []
            captions= []
            for condition in conditions.keys():                                  
                image_path = sbj_path + '/' + ch_type[ch]+conditions[condition]+over[ovr]+end
                    
                try:
                    fig, ax = plt.subplots()
                    img = mpimg.imread(image_path)
                    ax.imshow(img)
                    ax.set_axis_off()
        
                    figs.append(fig)
                    captions.append(condition)
                except:
                    
                    image_path = sbj_path + '/' + ch_type01[ch]+conditions[condition]+over[ovr]++end
                    
                    try:
                        fig, ax = plt.subplots()
                        img = mpimg.imread(image_path)
                        ax.imshow(img)
                        ax.set_axis_off()
            
                        figs.append(fig)
                        captions.append(condition)
                    except:
                        print('None of the worked')
                        
            report.add_figure(
                fig=figs, title=ovr, section = section,
                caption=captions
                )
            plt.close('all')

report.save(path.join(config.data_path, 'misc', 'systematic_comparison.html'), overwrite=True)

