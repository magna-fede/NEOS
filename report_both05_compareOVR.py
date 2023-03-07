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

conditions = {#"preica" : "_pre-ICA_",
             # "eog" : "eog_",
             # "var" : "var_",
             "both" : "both_"
             }

over = {"overweighted" : "ovrw_",
        "non-overweighted" : "",
        "onset overweighted" : "ovrwonset_"
        }

filtering =  {
        # '0.1': "01Hz",
        '0.5': "05Hz",
        # '1.0': "10Hz",
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

filt = '0.5'
condition = 'both'

for sbj_id in subjs:
    print(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0], 'Figures')

    
    figs = []
    captions= []    


    for ch in ch_type.keys():
        figs = []
        captions= []
        section = f'Subject {sbj_id} - ICA {ch}'
        for ovr in over.keys():
                                     
            image_path = sbj_path + '/' + ch_type[ch]+conditions[condition]+over[ovr]+filtering[filt]+end

            fig, ax = plt.subplots()
            img = mpimg.imread(image_path)
            ax.imshow(img)
            ax.set_axis_off()

            figs.append(fig)
            captions.append(condition+' '+filt+' Hz')

        report.add_figure(
            fig=figs, title=ovr, section = section,
            caption=captions
            )
        plt.close('all')

report.save(path.join(config.data_path, 'misc', 'OVR_comparison_both05.html'), overwrite=True)

