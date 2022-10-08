"""
===========
Config file for EOS with MEG
===========
"""

import os
from os import path

import sys

import numpy as np

###############################################################################

# IDs of subjects to process (SLURM and Grand-Average)
do_subjs = [5]

# path to acquired raw data
cbu_path = '/megdata/cbu/eyeonsemantics'

# path to data for pre-processing
data_path = '/imaging/hauk/users/fm02/MEG_EOS/data'

path_ET = '/imaging/hauk/users/fm02/MEG_EOS/ET_data'

if not path.isdir(data_path):  # create if necessary
    os.mkdir(data_path)

###############################################################################
# Mapping betwen filenames and subjects

map_subjects = {
    # 0: ('meg22_103', '220503'),
    # 1: ('trigger_test', '220715')  # pilot frequency sweep
    0 : ('meg22_156', '220720'), # first_pilot
    1 : ('meg22_165', '220805'), # first real participant
    2 : ('meg22_190', '221003'),
    3 : ('meg22_191', '221005'),
#    4 : ('meg22_192', '221006'), # participant did not complete experiment, very sleepy
    5 : ('meg22_193', '221007')
}

# which files to maxfilter and how to name them after sss
# [before maxfilter], [after maxfilter], [condition labels],
# [presentation/oddball frequencies]

sss_map_fnames = {
    # 0: (['pilot00_raw', 'pilot01_raw'],
    #     ['pilot00_sss_raw', 'pilot01_sss_raw']),
    # 1: (['trigger_test_raw'],
    #     ['trichk_sss_raw'])
    0 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    1 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    2 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    3 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']), 
    5 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),        
}       



###############################################################################
# Bad channels

# bad_channels = {
#     1: {'eeg': ['EEG028'],
#         'meg': ['MEG1123', 'MEG2223', 'MEG0813']}
# }


# create subject-specific data directories if necessary
for ss in map_subjects:
    # subject-specific sub-dir, e.g. maxfiltered raw data
    subj_dir = path.join(data_path, map_subjects[ss][0])

    if not path.isdir(subj_dir):
        print('Creating directory %s.' % subj_dir)
        os.mkdir(subj_dir)
        
    # Figures directory
    fig_dir = path.join(data_path, map_subjects[ss][0],
                         'Figures')  # subject figure dir
    if not path.isdir(fig_dir):
        print('Creating directory %s.' % fig_dir)
        os.mkdir(fig_dir)


# For subjects without clean ECG channel,
# use the following magnetometers in ICA (else specify '' to use ECG)
ECG_channels = {
    0 : '',
    1 : '',
    2 : '',
    3 : '',
}

# Artefact rejection thresholds
# for ICA, covariance matrix
reject = dict(grad=4e-10, mag=1e-11, eeg=1e-3)

###############################################################################
# ERPs

# artefact rejection thresholds for epoching
epo_reject = dict(grad=4e-10, mag=1e-11, eeg=1e-3)

# baseline in s
#epo_baseline = (-.2, 0.)

# epoch interval in s
#epo_t1, epo_t2 = -.2, .5

###############################################################################

###############################################################################
# Maxfilter etc.

# parameters for Neuromag maxfilter command
# Make sure to use Vectorview files!
MF = {
    'NM_cmd': '/imaging/local/software/neuromag/bin/util/maxfilter-2.2.12',
    'cal': '/neuro/databases_vectorview/sss/sss_cal.dat',
    'ctc': '/neuro/databases_vectorview/ctc/ct_sparse.fif',
    'st_duration': 10.,
    'st_correlation': 0.98,
    'origin': (0., 0., 0.045),
    'in': 8,
    'out': 3,
    'regularize': 'in',
    'frame': 'head',
    'mv': 'inter',
    'trans': 0}  # which file to use for -trans within subject

# for correcting EEG electrode positions
check_cmd = '/imaging/local/software/mne/mne_2.7.3/x86_64/\
MNE-2.7.3-3268-Linux-x86_64//bin/mne_check_eeg_locations \
--file %s --fix'

### FILTERING, EVENTS

# define the stim channel
stim_channel = 'STI101'

# bandpass filter frequencies
l_freq, h_freq = 0.1, 40.

raw_ICA_suff = 'ica_raw'


# EDF Label start trial
edf_start_trial = 'TRIGGER 94'
# EDF Label end trial
edf_end_trial = 'TRIGGER 95'

########################################################
# Edited for FPVS up to here
########################################################

### Epoching, Averaging

# stimulus projector delay
delay = 0.0345

# separate triggers for target detection and localiser tasks
event_id = {}

# Source Space
stc_morph = 'fsaverage'

# vertex size
src_spacing = 5
