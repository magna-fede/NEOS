#!/imaging/local/software/miniconda/envs/mne0.20/bin/python
"""
Synchronise ET and MEG data.
aka import ET data in MEG events.

author: federica.magnabosco@mrc-cbu.cam.ac.uk 
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

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

reject_criteria = config.epo_reject
flat_criteria = config.epo_flat

mne.viz.set_browser_backend("matplotlib")

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

sns.set_theme(context="notebook",
              style="white",
              font="sans-serif")

sns.set_style("ticks")


def synchronise(sbj_id, plot_events=False):
    """This function synchronise each MEG blockby incorporatin ET information.
    It is assumed that the trigger for start end of the sentence is shared between
        the MEG and the ET files.
    It requires data already preprocessed using pygaze output (list of dicts,
        where each element is a trial).
    It is assumed that MEG and ET have the same sampling rate (1000 Hz in our case).
    It is assumed that there is a unique identifier for each trials.
    It is assumed that each block contains exactly 80 trials.
    
    There are probably other assumptions underlying this function.
    
    NB: if you want to have the events not divided per block, then run 
        the fucntion synchronise_concat()
    """
    def trial_duration_ET(row):
        return int(row[config.edf_end_trial]) - int(row[config.edf_start_trial])
    
    # HEY! synchronise won't work with participant 0 and 1 bc of different
    # coding of triggers. You will need to create another synch for them.

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
            path.join(sbj_path, raw_stem_in[:-7] + 'sss_f_raw.fif'))
        
    print(f'Importing raw ET data {config.map_subjects[sbj_id][0][-3:]+".asc"}')
    edf = os.path.join(sbj_path_ET, config.map_subjects[sbj_id][0][-3:]+'.asc')

    # don't throw errors for non-UTF characters, replace them
    f = open(edf, 'r', errors='replace')
    raw_edf = f.readlines()
    f.close()

    print('Retrieving trigger values for start and end of a sentence')
    start_trial_ET = config.edf_start_trial
    end_trial_ET = config.edf_end_trial
     
    print('Retrieving trigger values for ssaving eye events in events MEG structure.')
    sac_trigger = config.sac_trig_value
    fix_trigger = config.fix_trig_value
    blk_trigger = config.blk_trig_value
    
    print('Store trigger values also as int for convenient access to MRI.')
    start_trigger_value = int(config.edf_start_trial.split()[1])
    end_trigger_value = int(config.edf_end_trial.split()[1])
 
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
    all_ET_trials = pd.DataFrame(trials)

    # Check that this is correct
    test_trials = pd.to_numeric(all_ET_trials[end_trial_ET]) - \
                    pd.to_numeric(all_ET_trials[start_trial_ET])
    assert all(test_trials) > 0, f"Trial ends before it starts at index: {test_trials[test_trials < 0].index[0]}"
    
    # load preprocessed ET file
    events_ET_file = os.path.join(
        sbj_path_ET, 'data_'+config.map_subjects[sbj_id][0][-3:]+'.P')
    with open(events_ET_file, 'rb') as f:
        all_events_ET = pickle.load(f)
    
    print('Getting information about fixated target words.')
    datafile_ET = os.path.join(sbj_path_ET,
                    'ET_info_'+config.map_subjects[sbj_id][0][-3:]+ '.csv')
    
    # get info about fixation durations on target word
    # only first-pass considered   
    data_ET = pd.read_csv(datafile_ET)
    data_ET = data_ET[data_ET['fixated'] == 1]         
            
    print(f'Reading raw file {sss_map_fnames}')

    for block, drf in enumerate(data_raw_files):
        
        print(f'Reading block {block+1}')
        print(f'Raw data: {drf}')
        data = mne.io.read_raw_fif(drf)
        
        # note that the eyetracking data is not divided per block, we artificially divide it now
        # as we know that we present exavtly 80 sentences per block, so we match them
        # with the 80 trials contained in the MEG block
        events_ET = all_events_ET[block*80:(block+1)*80]
        trials_ET = all_ET_trials[block*80:(block+1)*80].reset_index(drop=True)

        print('Reading events from STI101 channel')
        devents = mne.find_events(data,
                                  stim_channel='STI101',
                                  min_duration=0.002,
                                  consecutive=True)      
        
        # Create DataFrame with meg info about trigger ID
        # note, column "previous" is now useless, because we introduced a delay
        # with respect to the preceding trigger state in the presentation script
        # (while it was consecutive in previous versions)
        
        pd_events_meg = pd.DataFrame(devents)
        pd_events_meg.columns = ['time', 'previous', 'trigger']

        # get trial ids from the .edf data
        # in MEG trial ID is divided between two triggers
        # the first indicates the category 1-2-3-4-5
        # the second the specific ID
        # you can sum them for comparison with edf
        # eg., MEG: (2*100 + 32) == ET: (232)
        
        ids = []
    
        for trial in events_ET:
            ids.append(int(trial['events']['msg'][2][1].split()[2]))
        
        trials_ET['IDstim'] = ids
    
        print('Calculating each trials duration in MEG and ET.')
        # add column from meg for comparison
        trials_ET['meg_duration'] = 0
 
        # create new column in MEG events DataFrame with real trial ID
        for i in range(0, len(pd_events_meg)-1):
            if pd_events_meg.loc[i, 'trigger'] in [1, 2, 3, 4, 5]:
                pd_events_meg.loc[i, 'trig'] = pd_events_meg.loc[i, 'trigger']*100 + \
                    pd_events_meg.loc[i+1, 'trigger']
        
        del trial
        # get the MEG times
        for i, trial in enumerate(trials_ET['IDstim']):
            try:
                ix = pd_events_meg[pd_events_meg['trig'] == trial].index[0]
                # check that ID sentence trigger is preceded by start and followed by end of sentence trigger
                if ((pd_events_meg['trigger'].iloc[ix-1] == start_trigger_value) and
                        pd_events_meg['trigger'].iloc[ix+2] == end_trigger_value):
                    trials_ET.loc[i, 'meg_duration'] = pd_events_meg['time'].iloc[ix + \
                        2] - pd_events_meg['time'].iloc[ix-1]
            except:
                pass  
            
        # get the ET times
        trials_ET['eyelink_duration'] = trials_ET.apply(trial_duration_ET, axis=1)
        
        # calculate difference 
        trials_ET['difference'] = trials_ET['meg_duration'] - \
                                    trials_ET['eyelink_duration']
        
        print(f"Saving trials duration for ET and MEG and their difference \n \
              {path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + f'_trial_synch_block_{block+1}.csv')}")                            
        trials_ET.to_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                    f'_trial_synch_block_{block+1}.csv'), index=False)
    
        print(f'MAX trial duration difference (ms) {trials_ET["difference"].max()}')
        print(f'MIN trial duration difference (ms) {trials_ET["difference"].min()}')
        print(f'AVERAGE trial duration difference (ms) {trials_ET["difference"].mean()}')
        
        sns.displot(trials_ET['difference'], discrete=True).set_titles('Trial duration offset MEG-ET (ms)')
            
        plt.savefig(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_trial_offset_block_{block+1}.png'), format='png');
                   
        print('Include information about fixations on target from eye tracker.')
        
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
        
        print('Include information all events from eye tracker.')
        starttime_ET_trials = []
    
        for trial in events_ET:
            starttime_ET_trials.append(int(trial['events']['msg'][0][0]))
    
        # get sentence end and start times in MEG
        ix_start = np.where(devents == start_trigger_value)[0]
        ix_end = np.where(devents ==end_trigger_value)[0]
    
        startend = tuple(zip(ix_start, ix_end)) 
        
        events = dict()
        
        for event in ['Efix', 'Esac', 'Eblk']:
            events[event] = dict.fromkeys(['start', 'end', 'x', 'y'])
            for e in events[event].keys():
                events[event][e] = list()
                
        for i, trial in enumerate(events_ET):        
            for event in ['Efix', 'Esac', 'Eblk']:
                start_time = [t[0] for t in trial['events'][event]]
                end_time = [t[1] for t in trial['events'][event]]
                # for saccade by selecting last two elements
                # of each event, we get end position.
                # could be used when wanting to cut epoch to the
                # last saccade?
                
                # below is a list for each even of their x, y, start and end times
                x = [t[-2] for t in trial['events'][event]]
                y = [t[-1] for t in trial['events'][event]]
                start_time = [time - starttime_ET_trials[i] for time in start_time]
                end_time = [time - starttime_ET_trials[i] for time in end_time]
                
                # final is a list (N=400 trials) of lists (
                # N of each list is len(event) per trial)
                events[event]['start'].append(start_time)
                events[event]['end'].append(end_time) 
                events[event]['x'].append(x)              
                events[event]['y'].append(y) 
    
        print('Got all eye tracker events.')    
        events[fix_trigger] = events.pop('Efix')
        events[sac_trigger] = events.pop('Esac')
        events[blk_trigger] = events.pop('Eblk')
        
        print('Now adding all ET events as MEG triggers.')
        # note, triggers should be int that don't overlap with other events.
        # at the moment:
        print(f'Fixation start = {fix_trigger}.\nFixation end = {fix_trigger+1}.')
        print(f'Saccade start = {sac_trigger}.\nSaccade end = {sac_trigger+1}.')
        print(f'Blink start = {blk_trigger}.\nBlink end = {blk_trigger+1}.')
        
        events_with_eye = pd.DataFrame(devents)
        events_with_eye[[3, 4]] = np.nan
        
        for i, indices in enumerate(startend):
            # start_time is MEG time when a certain trial started
            start_time = devents[startend[i][0]][0]
            for event in events.keys():
                # we save timings for each event, for each trial
                starts = [start_time + time for time in events[event]['start'][i]]
                xs = events[event]['x'][i]
                ys = events[event]['y'][i]
                ends = [start_time + time for time in events[event]['end'][i]]
                for start, x, y in zip(starts, xs, ys):
                    events_with_eye = pd.concat([events_with_eye,
                                                 pd.DataFrame([
                                                     [start, 0, event, x, y]],
                                                     columns=[0,1,2,3,4])])
                for end, x, y in zip(ends, xs, ys):
                    events_with_eye = pd.concat([events_with_eye,
                                                 pd.DataFrame([
                                                     [end, 0, event+1, x, y]],
                                                     columns=[0,1,2,3,4])])
                    
        events_with_eye = events_with_eye.sort_values(by=[0]).reset_index(drop=True)      
        
        mne_events_with_eye = events_with_eye.drop([3,4], axis=1)
        events_with_eye.columns = ['time', 'previous', 'trigger', 'x', 'y']
        
        print(f"Save mne compatible events:\n only fixations on target : \
              {path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + f'_target_events_block_{block+1}.fif')}")
        mne.write_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_target_events_block_{block+1}.fif'), devents, overwrite=True)
        print(f"Save mne compatible events:\n all ET events : \
              {path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + f'_all_events_block_{block+1}.fif')}")
        mne.write_events(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                              f'_all_events_block_{block+1}.fif'), mne_events_with_eye, overwrite=True)   
        
        print(f"Save dataframe with x,y information for all events: \n \
              {path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + f'_all_events_xy_block_{block+1}.csv')}")
        
        events_with_eye.to_csv(path.join(sbj_path, config.map_subjects[sbj_id][0][-3:] + \
                                  f'_all_events_xy_block_{block+1}.csv'), index=False)
             
        # print(f"Uncorrected fixations plot {path.join(sbj_path, 'Figures', f'uncorrected_fixation_{i}.jpg')}")
        # event_dict = {'fixation': 901}
        # epochs = mne.Epochs(data, mne_events_with_eye, picks=['meg', 'eeg', 'eog'],
        #                     tmin=-0.3, tmax=0.7, event_id=event_dict,
        #                     reject=reject_criteria, flat=flat_criteria,
        #                     preload=True)
        # evoked = epochs['fixation'].average()
        # print('Do Plots')
        # fixation_fig = evoked.plot_joint(times=[0, .100, .167, .210, .266, .330, .430])
        
        # print('Save Plots')
        # for i, fig in zip(['EEG','MAG','GRAD'], fixation_fig):
        #     fname_fig = path.join(sbj_path, 'Figures', f'uncorrected_fixation_{sbj_id}_{block+1}.jpg')
        #     fig.savefig(fname_fig)    
        
            
        # print(f"Uncorrected saccades plot {path.join(sbj_path, 'Figures', f'uncorrected_saccade_{sbj_id}_{block+1}.jpg')}")    
        # event_dict = {'saccade': 801}
        # epochs = mne.Epochs(data, mne_events_with_eye, picks=['meg', 'eeg', 'eog'],
        #                     tmin=-0.1, tmax=0.7, event_id=event_dict,
        #                     reject=reject_criteria, flat=flat_criteria,
        #                     preload=True)
        # evoked = epochs['saccade'].average()
        # print('Do PLots.')
        # saccade_fig = evoked.plot_joint(times=[0, .100, .167, .210, .266, .330, .430])
        
        # print('Save Plots.')
        # for i, fig in zip(['EEG','MAG','GRAD'], saccade_fig):
        #     fname_fig = path.join(sbj_path, 'Figures', f'uncorrected_saccade_{sbj_id}_{block+1}.jpg')
        #     fig.savefig(fname_fig)        
        
        
# # get all input arguments except first
# if len(sys.argv) == 1:

#     sbj_ids = np.arange(0, len(config.map_subjects)) + 1

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     synchronise(ss)
        

