#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot only both method to compare overweighting procedures.

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
        
conditions = 'both'

over = {
        "non-overweighted" : "",
        "overweighted" : "_ovrw",
        "onset overweighted" : "_ovrwonset"
        }

for sbj_id in subjs:
    print(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0], 'Figures')

    for ch in ch_type.keys():
        figs = []
        captions= []
        for ovr in over.keys():
            section = f'Subject {sbj_id} - ICA {ch} {ovr}'

            image_path = sbj_path + '/' + ch_type[ch]+conditions+over[ovr]+end
                    
            try:
                fig, ax = plt.subplots()
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.set_axis_off()
    
                figs.append(fig)
                captions.append(ovr)
            except:
                
                image_path = sbj_path + '/' + ch_type01[ch]+conditions+over[ovr]+end
                
                try:
                    fig, ax = plt.subplots()
                    img = mpimg.imread(image_path)
                    ax.imshow(img)
                    ax.set_axis_off()
        
                    figs.append(fig)
                    captions.append(ovr)
                except:
                    print('None of the worked')
                    
        report.add_figure(
            fig=figs, title=ovr, section = section,
            caption=captions
            )
        plt.close('all')

report.save(path.join(config.data_path, 'notasystematic_comparison.html'), overwrite=True)

