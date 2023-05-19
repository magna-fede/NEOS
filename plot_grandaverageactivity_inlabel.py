#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:41:37 2023

@author: fm02
"""


import matplotlib.pyplot as plt

import mne

import os
from os import path

import numpy as np
import pandas as pd

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

conditions = ['Predictable', 'Unpredictable', 'Abstract', 'Concrete']


labels_path = path.join(config.data_path, "my_ROIs")
stc_path = path.join(config.data_path, "stcs")
ave_path = path.join(config.data_path, "AVE")

sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
           21,22,23,24,25,26,27,28,29,30]

ga = dict.fromkeys(conditions)


for key in ga.keys():
    ga[key] = pd.DataFrame()

for sbj_id in sbj_ids:
    for ev in conditions:
        activity = pd.read_csv(path.join(ave_path, "in_labels", f"{sbj_id}_{ev}_in_labels.csv"),
                               usecols=['lATL',
                                        'rATL',
                                        'PVA',
                                        'IFG',
                                        'AG',
                                        'PTC'])
        activity['ID'] = sbj_id
        ga[ev] = pd.concat([ga[ev], activity], axis=0)


ga_computed = dict()
ga_std = dict()
for ev in conditions:
    ga_computed[ev] = ga[ev].reset_index().groupby(['index']).mean().drop(['ID'], axis=1)
    ga_std[ev] = ga[ev].reset_index().groupby(['index']).std().drop(['ID'], axis=1)


times = np.linspace(-300, 696, 250)

for roi in ['lATL',
         'rATL',
         'PVA',
         'IFG',
         'AG',
         'PTC']:
    plt.figure()
    sns.lineplot(x=times, y=ga_computed['Predictable'][roi], label='predictable', )
    plt.fill_between(x=times, \
                  y1=(ga_computed['Predictable'][roi] - ga_std['Predictable'][roi]), \
                  y2=(ga_computed['Predictable'][roi] + ga_std['Predictable'][roi]), \
                  alpha=.1)
    sns.lineplot(x=times, y=ga_computed['Unpredictable'][roi], label='unpredictable')
    plt.fill_between(x=times, \
                  y1=(ga_computed['Unpredictable'][roi] - ga_std['Unpredictable'][roi]), \
                  y2=(ga_computed['Unpredictable'][roi] + ga_std['Unpredictable'][roi]), \
                  alpha=.1)
    plt.title(f"{roi}")
    plt.show()

for roi in ['lATL',
         'rATL',
         'PVA',
         'IFG',
         'AG',
         'PTC']:
    plt.figure()
    sns.lineplot(x=times, y=ga_computed['Concrete'][roi], label='Concrete', )
    plt.fill_between(x=times, \
                  y1=(ga_computed['Concrete'][roi] - ga_std['Concrete'][roi]), \
                  y2=(ga_computed['Concrete'][roi] + ga_std['Concrete'][roi]), \
                  alpha=.1)
    sns.lineplot(x=times, y=ga_computed['Abstract'][roi], label='Abstract')
    plt.fill_between(x=times, \
                  y1=(ga_computed['Abstract'][roi] - ga_std['Abstract'][roi]), \
                  y2=(ga_computed['Abstract'][roi] + ga_std['Abstract'][roi]), \
                  alpha=.1)
    plt.title(f"{roi}")
    plt.show()
plt.close('all')