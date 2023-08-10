#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:51:04 2023

@author: fm02
"""

import pandas as pd
import numpy as np
import mne
import seaborn as sns
import matplotlib.pyplot as plt


import sys
import os
from os import path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import mne
os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config
from my_eyeCA import apply_ica
from sklearn.preprocessing import OneHotEncoder
import h5py

sbj_id =1


a = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/coeftable_{sbj_id}_concpred.csv")

target = a[['channel', 'coefname','estimate', 'time']][a['basisname']=='targ']

times = target['time'].unique()

# plt.subplots()
# [sns.lineplot(y=target['estimate'][(target['coefname']=='(Intercept)') & (target['channel']==ch)], x=list(times)) for ch in range(1,65)]
# plt.subplots()
# [sns.lineplot(y=target['estimate'][(target['coefname']=='CU') & (target['channel']==ch)], x=list(times)) for ch in range(1,61)]
# plt.subplots()
# [sns.lineplot(y=target['estimate'][(target['coefname']=='AU') & (target['channel']==ch)], x=list(times)) for ch in range(1,61)]
# plt.subplots()

# [sns.lineplot(y=target['estimate'][(target['coefname']=='AP') & (target['channel']==ch)], x=list(times)) for ch in range(1,61)]
# plt.subplots()
# [sns.lineplot(y=target['estimate'][(target['coefname']=='CP') & (target['channel']==ch)], x=list(times)) for ch in range(1,61)]



fix = a[['channel', 'coefname','estimate', 'time']][a['basisname']=='fix']
# plt.subplots()
# [sns.lineplot(y=fix['estimate'][(fix['coefname']=='(Intercept)') & (fix['channel']==ch)], x=list(times)) for ch in range(1,61)]
# plt.subplots()
# [sns.lineplot(y=fix['estimate'][(fix['coefname']=='(Intercept)') & (fix['channel']==ch)], x=list(times)) for ch in range(1,61)]
# plt.title('All other fixations')
# plt.savefig('/home/fm02/MEG_NEOS/julia/general_fixations.png')
# plt.subplots()
# [sns.lineplot(y=target['estimate'][(target['coefname']=='(Intercept)') & (target['channel']==ch)], x=list(times)) for ch in range(0,60)]
# [sns.lineplot(y=target['estimate'][(target['coefname']=='(Intercept)') & (target['channel']==ch)], x=list(times)) for ch in range(1,61)]
# plt.title('Average target fixations')
# plt.savefig('/home/fm02/MEG_NEOS/julia/intercept_target_fixations.png')



meta = pd.read_csv('/imaging/hauk/users/fm02/MEG_NEOS/stim/meg_metadata.csv', header=0)

def ovr_sub(ovr):
    if ovr in ['nover', 'novr', 'novrw']:
        ovr = ''
    elif ovr in ['ovrw', 'ovr', 'over', 'overw']:
        ovr = '_ovrw'
    elif ovr in ['ovrwonset', 'ovrons', 'overonset']:
        ovr = '_ovrwonset'
    return ovr


subject = str(sbj_id)
sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
bad_eeg = config.bad_channels_all[sbj_id]['eeg']

ovr = config.ovr_procedure[sbj_id]
ovr = ovr_sub(ovr)

raw = mne.io.read_raw(path.join(sbj_path, "block1_sss_raw.fif"))

eeg = [fix['estimate'][(fix['coefname']=='(Intercept)') & (fix['channel']==ch)] for ch in range(1,65)]

meg = [fix['estimate'][(fix['coefname']=='(Intercept)') & (fix['channel']==ch)] for ch in range(65,371)]
eeg = np.array(eeg)
meg = np.array(meg)

data = np.concatenate(([eeg, meg]))

raw.resample(250)

raw.pick_types(meg=True, eeg=True)

info = raw.info

epoch = mne.EvokedArray(data, raw.info, tmin=-0.15)
epoch.plot()

epoch.plot_joint()

# target fixation [predicable, concrete]
eeg = [target['estimate'][(target['coefname']=='(Intercept)') & (target['channel']==ch)] for ch in range(1,65)]

meg = [target['estimate'][(target['coefname']=='(Intercept)') & (target['channel']==ch)] for ch in range(65,371)]
eeg = np.array(eeg)
meg = np.array(meg)

data = np.concatenate(([eeg, meg]))

eeg = [target['estimate'][(target['coefname']=='Predictability: 1.0') & (target['channel']==ch)] for ch in range(1,65)]

meg = [target['estimate'][(target['coefname']=='Predictability: 1.0') & (target['channel']==ch)] for ch in range(65,371)]
eeg = np.array(eeg)
meg = np.array(meg)

data = np.concatenate(([eeg, meg]))



bad_eeg = config.bad_channels_all[sbj_id]['eeg']

epoch.info['bads'] = bad_eeg
epoch.pick_types(meg=True, eeg=True, exclude='bads')

method='eLORETA'
inv_suf='shrunk_dropbads'
inv_fname = path.join(sbj_path, subject + f'_EEGMEG{inv_suf}-inv.fif')
inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)

snr = 3.0
lambda2 = 1.0 / snr**2

epoch.set_eeg_reference(projection=True)

stc = mne.minimum_norm.apply_inverse(epoch, inverse_operator,
                                      lambda2, method=method,
                                      pick_ori=None, verbose=True)

surfer_kwargs = dict(
    hemi="both",
    subjects_dir=config.subjects_dir,
    clim=dict(kind="value", lims=[8, 12, 15]),
    views="lateral",
    time_unit="s",
    size=(800, 800),
    smoothing_steps=10,
)
brain = stc.plot(**surfer_kwargs)


labels_path = path.join(config.data_path, "my_ROIs")

lATL = mne.read_label(path.join(labels_path, 'l_ATL_fsaverage-lh.label'),
                       subject='fsaverage')
lATL.name='lATL'
rATL = mne.read_label(path.join(labels_path, 'r_ATL_fsaverage-rh.label'),
                      subject='fsaverage')
rATL.name='rATL'
PVA = mne.read_label(path.join(labels_path, 'PVA_fsaverage-lh.label'),
                      subject='fsaverage')
PVA.name='PVA'
IFG = mne.read_label(path.join(labels_path, 'IFG_fsaverage-lh.label'),
                      subject='fsaverage')
IFG.name='IFG'
AG = mne.read_label(path.join(labels_path, 'AG_fsaverage-lh.label'),
                      subject='fsaverage')
AG.name='AG'
PTC = mne.read_label(path.join(labels_path, 'PTC_fsaverage-lh.label'),
                      subject='fsaverage')
PTC.name='PTC'
   
fname_fsaverage_src = path.join(config.subjects_dir,
                                subject,
                                'bem', 
                                f'{subject}_5-src.fif')
src = mne.read_source_spaces(fname_fsaverage_src)

rois = [lATL,
        rATL, 
        PVA,
        IFG,
        AG,
        PTC]

rois_sub = mne.morph_labels(rois, subject, 'fsaverage', config.subjects_dir)

roi_activity = []

for roi in rois_sub:
    a = stc.in_label(roi)
    roi_activity.append(stc.extract_label_time_course(roi, src, mode='mean'))        
 
for roi in roi_activity:

    fig, ax = plt.subplots(1);
    ax.plot(list(times), roi.T, 'k', linewidth=2);
    ax.set(xlabel='Time (ms)', ylabel='Source amplitude')


######
eeg_pred = [target['estimate'][(target['coefname']=='Predictable') & (target['channel']==ch)] for ch in range(1,65)]
meg_pred = [target['estimate'][(target['coefname']=='Predictable') & (target['channel']==ch)] for ch in range(65, 371)]

eeg_unpred = [target['estimate'][(target['coefname']=='Unpredictable') & (target['channel']==ch)] for ch in range(1,65)]
meg_unpred = [target['estimate'][(target['coefname']=='Unpredictable') & (target['channel']==ch)] for ch in range(65, 371)]


data_pred = np.concatenate(([eeg_pred, meg_pred]))
data_unpred = np.concatenate(([eeg_unpred, meg_unpred]))

evoked_pred = mne.EvokedArray(data_pred, info, tmin=-0.15)
evoked_unpred = mne.EvokedArray(data_unpred, info, tmin=-0.15)

contrast = mne.combine_evoked([evoked_pred, evoked_unpred], weights=[1, -1])
contrast.plot_joint()

epoch.plot()

epoch.plot_joint()

####
eeg_pred = [target['estimate'][(target['coefname']=='AP') & (target['channel']==ch)] for ch in range(1,65)]
meg_pred = [target['estimate'][(target['coefname']=='AP') & (target['channel']==ch)] for ch in range(65, 371)]

eeg_unpred = [target['estimate'][(target['coefname']=='CP') & (target['channel']==ch)] for ch in range(1,65)]
meg_unpred = [target['estimate'][(target['coefname']=='CP') & (target['channel']==ch)] for ch in range(65, 371)]


data_pred = np.concatenate(([eeg_pred, meg_pred]))
data_unpred = np.concatenate(([eeg_unpred, meg_unpred]))


evoked_pred = mne.EvokedArray(data_pred, info, tmin=-0.15)
evoked_unpred = mne.EvokedArray(data_unpred, info, tmin=-0.15)

predictable = mne.combine_evoked([evoked_pred, evoked_unpred], weights=[1, 1])

eeg_pred = [target['estimate'][(target['coefname']=='AU') & (target['channel']==ch)] for ch in range(1,65)]
meg_pred = [target['estimate'][(target['coefname']=='AU') & (target['channel']==ch)] for ch in range(65, 371)]

eeg_unpred = [target['estimate'][(target['coefname']=='CU') & (target['channel']==ch)] for ch in range(1,65)]
meg_unpred = [target['estimate'][(target['coefname']=='CU') & (target['channel']==ch)] for ch in range(65, 371)]


data_pred = np.concatenate(([eeg_pred, meg_pred]))
data_unpred = np.concatenate(([eeg_unpred, meg_unpred]))


evoked_pred = mne.EvokedArray(data_pred, info, tmin=-0.15)
evoked_unpred = mne.EvokedArray(data_unpred, info, tmin=-0.15)

unpredictable = mne.combine_evoked([evoked_pred, evoked_unpred], weights=[1, 1])

contrast = mne.combine_evoked([predictable, unpredictable], weights=[1, -1])


contrast.plot_joint()

#######
a = pd.read_csv(f"/imaging/hauk/users/fm02/MEG_NEOS/jl_evts/effect_{sbj_id}_pred.csv")

target = a[['channel', 'Predictability','yhat', 'time']][a['basisname']=='targ']
fix = a[['channel', 'Predictability','yhat', 'time']][a['basisname']=='fix']

eeg_P = [target['yhat'][(target['Predictability']=='Predictable') & (target['channel']==ch)] for ch in range(1,65)]

meg_P = [target['yhat'][(target['Predictability']=='Predictable') & (target['channel']==ch)] for ch in range(65,371)]
eeg_P = np.array(eeg_P)
meg_P = np.array(meg_P)

data_P = np.concatenate(([eeg_P, meg_P]))
evoked_P = mne.EvokedArray(data_P, info, tmin=-0.15)

eeg_U = [target['yhat'][(target['Predictability']=='Unpredictable') & (target['channel']==ch)] for ch in range(1,65)]

meg_U = [target['yhat'][(target['Predictability']=='Unpredictable') & (target['channel']==ch)] for ch in range(65,371)]
eeg_U = np.array(eeg_U)
meg_U = np.array(meg_U)

data_U = np.concatenate(([eeg_U, meg_U]))
evoked_U = mne.EvokedArray(data_U, info, tmin=-0.15)

contrast = mne.combine_evoked([evoked_P, evoked_U], weights=[1, -1])


contrast.plot_joint()


