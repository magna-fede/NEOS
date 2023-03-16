#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:28:37 2023

@author: fm02
"""
# import subprocess

# # This is our shell command, executed by Popen.

# p = subprocess.Popen("source ~/.cshrc", stdout=subprocess.PIPE, shell=True)

# print(p.communicate())

from IPython import get_ipython
ipython = get_ipython()

from julia.api import Julia
jl = Julia(compiled_modules=False)
import julia
from julia import Main
from julia import Unfold
from julia import DataFrames
from julia import  Pandas
from julia import CSV


#%load_ext julia.magic
ipython.run_line_magic("load_ext", "julia.magic")

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


all_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_all_events.fif'))
    
target_evts = mne.read_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_target_events.fif'))

all_evts = pd.DataFrame(all_evts, columns=['time','useless','trigger'])

fixations = all_evts[all_evts['trigger']==901]

targ_n_fix = pd.concat([pd.DataFrame(target_evts, columns=['time','useless','trigger']), fixations])

targ_n_fix = targ_n_fix.drop_duplicates(subset=['time'], keep='first')

targ_n_fix['trigger'].value_counts()
targ_n_fix = targ_n_fix.sort_values(by=['time'])

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
    
t_ds = targ_n_fix.copy()
t_ds['time'] = (t_ds['time']/4).apply(np.floor).astype(int) # careful this is must be same sampling rate of data


ev = t_ds[t_ds['trigger'].isin([901,999])].reset_index(drop=True)

ev = ev.rename(columns={'time': 'latency'})
ev['intercept'] = 1
ev['type'] = 'fixation'

ev['type'].loc[ev['trigger']==999] = 'target'

ev['ConcM'] = np.nan
ev['Pred'] = np.nan

ev['ConcM'].loc[ev['trigger']==999] = metadata['ConcM'].values
ev['Pred'].loc[ev['trigger']==999] = metadata['Sim'].values

ev[['ConcM', 'Pred']] = sc.fit_transform(ev[['ConcM', 'Pred']])

Main.eval("using StatsModels")
formula  = Main.eval("@formula 0~1+ConcM+Pred")

bfTarget = Unfold.firbasis(τ=(-0.2,.6),sfreq=250,name="target")
bfFixation = Unfold.firbasis(τ=(-0.1,.3),sfreq=250,name="fixation")

# bfDict = %julia Dict("target"=>($formula, $bfTarget),
#                      "fixation"=>($formula, $bfFixation))

bfDict = ipython.run_line_magic("julia", 'Dict("target"=>($formula, $bfTarget), "fixation"=>($formula, $bfFixation))')


# evts = %julia DataFrames.DataFrame(Pandas.DataFrame($ev))
evts = ipython.run_line_magic("julia", "DataFrames.DataFrame(Pandas.DataFrame($ev))")

# res = %julia Unfold.fit(Unfold.UnfoldModel,$bfDict,$evts,$norm_data,solver=(x,y) -> Unfold.solver_default(x,y;stderror=true),eventcolumn="type")
res = ipython.run_line_magic("julia", 'Unfold.fit(Unfold.UnfoldModel,$bfDict,$evts,$norm_data,solver=(x,y) -> Unfold.solver_default(x,y;stderror=true),eventcolumn="type")')

# results = %julia coeftable($res)
results = ipython.run_line_magic("julia", "coeftable($res)")

# %julia something.($results, missing) |> CSV.write("/home/fm02/misc/coeftableoveralp_norm_$($sbj_id)_all.csv")
ipython.run_line_magic("julia", 'something.($results, missing) |> CSV.write("/home/fm02/misc/coeftableoveralp_norm_$($sbj_id)_all.csv")')


# df2 = pd.read_csv("/home/fm02/misc/coeftableoveralp_norm_1.csv")
# df2['basisname'].head(5)
# df2['basisname'].head(-55)
# df2['coefname'].value_counts
# df2['coefname'].value_counts()
# df2.columns
# sns.lineplot(x = df2['time'], y=df2['estimate'], hue=df2['coefname']);
# import seaborn as sns
# sns.lineplot(x = df2['time'], y=df2['estimate'], hue=df2['coefname']);
# d.mean(axis=0)
# d.mean(axis=0).shape
# d.mean(axis=1).shape
# d.mean(axis=1)
# sns.lineplot(x = df2['time'][df2['channel']==1], y=df2['estimate'][df2['channel']==1], hue=df2['coefname'][df2['channel']==1]);
# sns.lineplot(x = df2['time'][df2['channel']==2], y=df2['estimate'][df2['channel']==1], hue=df2['coefname'][df2['channel']==1]);
# sns.lineplot(x = df2['time'][df2['channel']==2], y=df2['estimate'][df2['channel']==2], hue=df2['coefname'][df2['channel']==2]);
# sns.lineplot(x = df2['time'][df2['channel']==3], y=df2['estimate'][df2['channel']==3], hue=df2['coefname'][df2['channel']==3]);
# sns.lineplot(x = df2['time'][df2['channel']==4], y=df2['estimate'][df2['channel']==4], hue=df2['coefname'][df2['channel']==4]);
# d.mean(axis=1)
# df2['channel'].values
# sns.lineplot(x = df2['time'][df2['channel']==350], y=df2['estimate'][df2['channel']==350], hue=df2['coefname'][df2['channel']==350]);
# plt.show()
# import matplotlib.pyplot as okt
# import matplotlib.pyplot as plt
# sns.lineplot(x = df2['time'][df2['channel']==350], y=df2['estimate'][df2['channel']==350], hue=df2['coefname'][df2['channel']==350]);
# sns.lineplot(x = df2['time'][df2['channel']==250], y=df2['estimate'][df2['channel']==250], hue=df2['coefname'][df2['channel']==250]);
# sns.lineplot(x = df2['time'][df2['channel']==150], y=df2['estimate'][df2['channel']==150], hue=df2['coefname'][df2['channel']==150]);
# sns.lineplot(x = df2['time'][df2['channel']==32], y=df2['estimate'][df2['channel']==32], hue=df2['coefname'][df2['channel']==32]);
# sns.lineplot(x = df2['time'][df2['channel']==28], y=df2['estimate'][df2['channel']==28], hue=df2['coefname'][df2['channel']==28]);
