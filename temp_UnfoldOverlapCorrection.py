#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:28:37 2023

@author: fm02
"""

from julia.api import Julia
jl = Julia(compiled_modules=False)
import julia
from julia import Main
from julia import Unfold
from julia import DataFrames
from julia import  Pandas
from julia import CSV


%load_ext julia.magic

import sys
import os
from os import path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import mne
os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

sbj_id = 1

meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv', header=0)

sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
bad_eeg = config.bad_channels[sbj_id]['eeg']

ovr = config.ovr_procedure
sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
bad_eeg = config.bad_channels[sbj_id]['eeg']

if ovr[sbj_id] == 'ovrons':
    over = '_ovrwonset'
elif ovr[sbj_id] == 'ovr':
    over = '_ovrw'
elif ovr[sbj_id] == 'novr':
    over = ''
condition = 'both'

raw_test = []   

for i in range(1,6):
    raw_test.append(mne.io.read_raw(path.join(sbj_path, f"block{i}_sss_f_ica{over}_{condition}_raw.fif")))

raw_test= mne.concatenate_raws(raw_test)
raw_test.load_data()
raw_test.info['bads'] = bad_eeg

raw_test.interpolate_bads(reset_bads=True)
raw_test.filter(l_freq=0.5, h_freq=None)

target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_target_events.fif'))

rows = np.where(target_evts[:,2]==999)[0]

event_dict = {'FRP': 999}

tmin, tmax = -.3, .7

    # regular epoching

epochs = mne.Epochs(raw_test, target_evts, event_dict, tmin=tmin, tmax=tmax,
                    reject=None, preload=True)

metadata = pd.DataFrame(columns=meta.columns)

for row in rows: 
    index = target_evts[row-2, 2]*100 + target_evts[row-1, 2]
    metadata = pd.concat([metadata,
                          meta[meta['ID']==index]])

epochs.metadata = metadata
metadata

data = raw_test.copy().resample(250)
d = data.get_data(picks=['eeg','meg'])

norm_data = d

sc = StandardScaler()

norm_data = sc.fit_transform(norm_data.T).T
    
t_ds = target_evts.copy()
t_ds[:,0] = t_ds[:,0]/4 # careful this is must be same sampling rate of data

events = pd.DataFrame(columns=['latency', 'type', 'intercept', 'ConcM', 'Pred'])
events['latency'] = t_ds[np.where(t_ds[:,2]==999)[0], 0]
events['intercept'] = 1
events['type'] = 'fixation'

events['ConcM'] = metadata['ConcM'].values
events['Pred'] = metadata['Sim'].values

Main.eval("using StatsModels")
formula  = Main.eval("@formula 0~1+ConcM+Ped")

basisfunction = Unfold.firbasis(τ=(-0.4,.8),sfreq=50,name="stimulus")

bfDict = %julia Dict(Any=>($formula, $basisfunction))

evts = %julia DataFrames.DataFrame(Pandas.DataFrame($events))


res = Unfold.fit(Unfold.UnfoldModel,bfDict,evts,d)

results = %julia coeftable($res)
%julia something.($results, missing) |> CSV.write("/home/fm02/misc/coeftableoveralp_norm_$($sbj_id).csv")

df2 = pd.read_csv("/home/fm02/misc/coeftableoveralp_norm_1.csv")
df2['basisname'].head(5)
df2['basisname'].head(-55)
df2['coefname'].value_counts
df2['coefname'].value_counts()
df2.columns
sns.lineplot(x = df2['time'], y=df2['estimate'], hue=df2['coefname']);
import seaborn as sns
sns.lineplot(x = df2['time'], y=df2['estimate'], hue=df2['coefname']);
d.mean(axis=0)
d.mean(axis=0).shape
d.mean(axis=1).shape
d.mean(axis=1)
sns.lineplot(x = df2['time'][df2['channel']==1], y=df2['estimate'][df2['channel']==1], hue=df2['coefname'][df2['channel']==1]);
sns.lineplot(x = df2['time'][df2['channel']==2], y=df2['estimate'][df2['channel']==1], hue=df2['coefname'][df2['channel']==1]);
sns.lineplot(x = df2['time'][df2['channel']==2], y=df2['estimate'][df2['channel']==2], hue=df2['coefname'][df2['channel']==2]);
sns.lineplot(x = df2['time'][df2['channel']==3], y=df2['estimate'][df2['channel']==3], hue=df2['coefname'][df2['channel']==3]);
sns.lineplot(x = df2['time'][df2['channel']==4], y=df2['estimate'][df2['channel']==4], hue=df2['coefname'][df2['channel']==4]);
d.mean(axis=1)
df2['channel'].values
sns.lineplot(x = df2['time'][df2['channel']==350], y=df2['estimate'][df2['channel']==350], hue=df2['coefname'][df2['channel']==350]);
plt.show()
import matplotlib.pyplot as okt
import matplotlib.pyplot as plt
sns.lineplot(x = df2['time'][df2['channel']==350], y=df2['estimate'][df2['channel']==350], hue=df2['coefname'][df2['channel']==350]);
sns.lineplot(x = df2['time'][df2['channel']==250], y=df2['estimate'][df2['channel']==250], hue=df2['coefname'][df2['channel']==250]);
sns.lineplot(x = df2['time'][df2['channel']==150], y=df2['estimate'][df2['channel']==150], hue=df2['coefname'][df2['channel']==150]);
sns.lineplot(x = df2['time'][df2['channel']==32], y=df2['estimate'][df2['channel']==32], hue=df2['coefname'][df2['channel']==32]);
sns.lineplot(x = df2['time'][df2['channel']==28], y=df2['estimate'][df2['channel']==28], hue=df2['coefname'][df2['channel']==28]);
