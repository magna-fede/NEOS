#!/imaging/local/software/miniconda/envs/mne0.20/bin/python
"""
Apply ICA for FPVS Frequency Sweep.

Decompostion computed in FPVS_Compute_ICA.py
Based on Fiff_Apply_ICA.py.
==========================================
OH, modified October 2019
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
import matplotlib.pyplot as plt

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

reload(config)

def trial_duration_ET(row):
    return int(row['TRIGGER 95']) - int(row['TRIGGER 94'])


def synchronise(sbj_id):
    # HEY! synchronise won't work with participant 0-1 bc of different
    # coding of triggers. use first_participant.ipynb for them.

    # path to participant folder
    sbj_path = path.join(config.data_path, config.map_subjects[sbj_id][0])
    sbj_path_ET = path.join(
        config.path_ET, config.map_subjects[sbj_id][0][-3:])

    # raw-filename mappings for this subject
    tmp_fnames = config.sss_map_fnames[sbj_id][1]

    # only use files for correct conditions
    sss_map_fnames = []
    for sss_file in tmp_fnames:
        sss_map_fnames.append(sss_file)

    data_raw_files = []

    for raw_stem_in in sss_map_fnames:
        data_raw_files.append(
            path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_ica_raw.fif'))
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf))

    print('Concatenating data')
    data = mne.concatenate_raws(data)

    print('Reading events from STI101 channel')
    devents = mne.find_events(data,
                              stim_channel='STI101',
                              min_duration=0.002,
                              consecutive=True)

    print(f'Importing raw ET data {config.map_subjects[sbj_id][0][-3:]+".asc"}')
    edf = os.path.join(sbj_path_ET, config.map_subjects[sbj_id][0][-3:]+'.asc')

    # don't throw errors for non-UTF characters, replace them
    f = open(edf, 'r', errors='replace')
    raw_edf = f.readlines()
    f.close()

    print('Retrieving trigger values for start and end of a sentence')
    start_trial_ET = config.edf_start_trial
    end_trial_ET = config.edf_end_trial

    trials = dict()
    trials[start_trial_ET] = []
    trials[end_trial_ET] = []

    print('Getting start-end time of each trial for the ET')
    # this will be used for comparison with respect to MEG trial duration
    for line in raw_edf:
        if start_trial_ET in line:
            trials[start_trial_ET].append(line.split()[1])
        elif end_trial_ET in line:
            trials[end_trial_ET].append(line.split()[1])

    # transform trials to DataFrame
    pd_trials = pd.DataFrame(trials)

    # load preprocessed ET file
    ############## NEED TO IMPROVE THIS!!!! ########################
    events_ET_file = os.path.join(
        sbj_path_ET, 'data_'+config.map_subjects[sbj_id][0][-3:]+'.P')
    with open(events_ET_file, 'rb') as f:
        events_ET = pickle.load(f)

    # Create DataFrame with meg info about trigger ID
    # note, previous should now be useless, because we introduced a delay
    # with respect to the preceding trigger state
    # (while it was consecutive in previous versions)
    pd_events_meg = pd.DataFrame(devents)
    pd_events_meg.columns = ['time', 'previous', 'trigger']

    # get trial ids from the .edf data
    ids = []

    for trial in events_ET:
        ids.append(int(trial['events']['msg'][2][1].split()[2]))

    pd_trials['IDstim'] = ids

    print('Calculating each trial')
    pd_trials['meg_duration'] = 0

    for i in range(0, len(pd_events_meg)-1):
        if pd_events_meg.loc[i, 'trigger'] in [1, 2, 3, 4, 5]:
            pd_events_meg.loc[i, 'trig'] = pd_events_meg.loc[i, 'trigger']*100 + \
                pd_events_meg.loc[i+1, 'trigger']

    for i, trial in enumerate(pd_trials['IDstim']):
        try:
            ix = pd_events_meg[pd_events_meg['trig'] == trial].index[0]

            if ((pd_events_meg['trigger'].iloc[ix-1] == 94) and
                    pd_events_meg['trigger'].iloc[ix+2] == 95):
                pd_trials.loc[i, 'meg_duration'] = pd_events_meg['time'].iloc[ix + \
                    2] - pd_events_meg['time'].iloc[ix-1]
        except:
            pass

    pd_trials['eyelink_duration'] = pd_trials.apply(trial_duration_ET, axis=1)

    pd_trials['difference'] = pd_trials['meg_duration'] - \
                                pd_trials['eyelink_duration']
                                
    pd_trials.to_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                               '_trial_synch.csv'), index=False)

    print(f'MAX trial duration difference (ms) {pd_trials["difference"].max()}')
    print(f'MIN trial duration difference (ms) {pd_trials["difference"].min()}')
    print(f'AVERAGE trial duration difference (ms) {pd_trials["difference"].mean()}')
    
    sns.histplot(pd_trials['difference'], discrete=True).set_title( \
                    'Trial duration offset MEG-ET (ms');
        
    plt.savefig(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_trial_offset.png'), format='png');
        
    datafile_ET = os.path.join(sbj_path_ET,
                    'ET_info_'+config.map_subjects[sbj_id][0][-3:]+'.csv')
    
    data_ET = pd.read_csv(datafile_ET)
    data_ET = data_ET[data_ET['fixated'] == 1]
    
    trigger_FRP = 999
    for id_stim in data_ET['IDStim']:
        stim_index = pd_events_meg['trig'][pd_events_meg['trig']==id_stim].index.values
        if stim_index.size > 0:
            timing = pd_events_meg['time'][stim_index-1].values[0] + \
                data_ET['time'][data_ET['IDStim']==id_stim].values[0]
            to_insert = np.array([timing, 0, trigger_FRP])
            pos_insertion = np.where(devents[:,0] == \
                        pd_events_meg['time'][stim_index-1].values[0])[0] + 3
            devents = np.insert(devents, pos_insertion, to_insert, axis=0)
    
    np.save(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_FIX_eve.npy'), devents)
    mne.write_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                          '_FIX_eve.fif'), devents, overwrite=True)
        
# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, 18) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    synchronise(ss)
        

