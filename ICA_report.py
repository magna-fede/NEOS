#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a HTML report where 
    - for each participant
        - for each condition
            we plot comparison of different ICAs correction.

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

report = mne.Report(title='Various ICA pipelines')

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

subjs = config.do_subjs


opticat = {
     'eeg': 'FRP_all_EEG_opticat.png',                    
     'grad': 'FRP_all_GRAD_opticat.png',
     'mag': 'FRP_all_MAG_opticat.png',   
     'predictable': 'FRP_predictable_opticat.png',
     'unpredictable': 'FRP_unpredictable_opticat.png',
     'predictability': 'FRP_predictability_opticat.png',
    # 'FRP_abstract_opticat.png',                   
    # 'FRP_concrete_opticat.png',                   
    # 'FRP_concreteness_opticat.png',  
     }             
    

unfiltered_opticat = {
     'eeg': 'FRP_all_EEG_opticat_unfiltered.png',
     'grad': 'FRP_all_GRAD_opticat_unfiltered.png',
     'mag': 'FRP_all_MAG_opticat_unfiltered.png',
     'predictable': 'FRP_predictable_opticat_unfiltered.png',
     'unpredictable': 'FRP_unpredictable_opticat_unfiltered.png',
     'predictability':  'FRP_predictability_opticat_unfiltered.png',
    # 'FRP_abstract_opticat_unfiltered.png',        
    # 'FRP_concrete_opticat_unfiltered.png',        
    # 'FRP_concreteness_opticat_unfiltered.png',    
     }
    

unfiltered_opticat_raw = {
     'eeg': 'FRP_all_EEG_opticat_unfiltered_raw.png',
     'grad': 'FRP_all_GRAD_opticat_unfiltered_raw.png',
     'mag': 'FRP_all_MAG_opticat_unfiltered_raw.png',
     'predictable': 'FRP_predictable_opticat_unfiltered_raw.png',
     'unpredictable': 'FRP_unpredictable_opticat_unfiltered_raw.png',
     'predictability':  'FRP_predictability_opticat_unfiltered_raw.png',
    # 'FRP_abstract_opticat_unfiltered.png',        
    # 'FRP_concrete_opticat_unfiltered.png',        
    # 'FRP_concreteness_opticat_unfiltered.png',    
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
    

uncorrected = {
     'eeg': 'uncorrected_fixation_EEG.png',
     'grad': 'uncorrected_fixation_GRAD.png',
     'mag': 'uncorrected_fixation_MAG.png',
    # 'uncorrected_saccade_EEG.png',
    # 'uncorrected_saccade_GRAD.png',
    # 'uncorrected_saccade_MAG.png',
    }

ica_per_block = {
     'eeg': 'FRP_all_EEG_ica_perblock_raw.png',
     'grad': 'FRP_all_GRAD_ica_perblock_raw.png',
     'mag': 'FRP_all_MAG_ica_perblock_raw.png',
     'predictable': 'FRP_predictable_ica_perblock_raw.png',
     'unpredictable': 'FRP_unpredictable_ica_perblock_raw.png',
     'predictability':  'FRP_predictability_ica_perblock_raw.png'
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

    figs = []
    captions= []    
    for unc in uncorrected.values():
        section = f'Subject {sbj_id} - uncorrected'

        image_path = sbj_path + '/' + unc
            
        try:
            fig, ax = plt.subplots()
            img = mpimg.imread(image_path)
            ax.imshow(img)

            figs.append(fig)
            captions.append(unc)
        except:
            print(f'{image_path}.png does not exist')
        
    report.add_figure(
        fig=figs, title='Uncorrected', section = section,
        caption=captions
        )
    plt.close('all')
    
    single_group = conditions['standard'].keys()
    for group in single_group:  
        section = f'Subject {sbj_id} - ICA {group}'
        figs = []
        captions= []
        for condition in conditions:
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

report.save(path.join(config.data_path, 'misc', 'methodsday_custom_figures.html'), overwrite=True)

