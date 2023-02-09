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

subjs = config.do_subjs

end = ".png"

over = {"ovrw" : "overweight",
        "novrw" : "NOoverweight"
        }

conditions = {"preica" : "_pre-ICA_",
             "eog" : "_eog_",
             "var" : "_variance_",
             "both" : "_both_"
             }
ch_type = {
     'eeg': 'FRP_all_EEG__',                    
     'grad': 'FRP_all_GRAD__',
     'mag': 'FRP_all_MAG__',   
     }             
    


    

standard = {
     'eeg': 'FRP_all_EEG_normalica.png',
     'grad': 'FRP_all_GRAD_normalica.png',
     'mag': 'FRP_all_MAG_normalica.png',     
     'predictable': 'FRP_predictable_normalica.png',
     'unpredictable': 'FRP_unpredictable_normalica.png',
     'predictability': 'FRP_predictability_contrast_normalica.png',  
    # 'FRP_abstract_normalica.png',                 
    # 'FRP_concrete_normalica.png',                 
    # 'FRP_concreteness_normalica.png',          
     }


conditions = dict({
                  'standard': standard, 
                  'opticat': opticat,
                  'unfiltered_opticat': unfiltered_opticat,
                  'unfiltered_opticat_raw': unfiltered_opticat_raw,  
                  'ica_per_block' : ica_per_block
                  })

for sbj_id in subjs:
    print(sbj_id)
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0], 'Figures')
    
    single_group = over.items()
    for ch in ch_types
    for group in single_group:  
        section = f'Subject {sbj_id} - ICA {group}'
        figs = []
        captions= []
        for condition in conditions.keys():
            image_path = sbj_path + '/' + conditions[condition][group]
            
            try:
                fig, ax = plt.subplots()
                img = mpimg.imread(image_path)
                ax.imshow(img)
    
                figs.append(fig)
                captions.append(condition)
            except:
                print(f'{image_path}.png does not exist')
        
        report.add_figure(
            fig=figs, title=condition, section = section,
            caption=captions
            )
        plt.close('all')

report.save(path.join(config.data_path, 'misc', 'systematic_comparison.html'), overwrite=True)

