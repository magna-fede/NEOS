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
do_subjs = [
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

# do_subjs = [21]
# path to acquired raw data
cbu_path = '/megdata/cbu/eyeonsemantics'

# path to data for pre-processing
data_path = '/imaging/hauk/users/fm02/MEG_NEOS/data'
dataold_path = '/imaging/hauk/users/fm02/MEG_NEOS/data_old'


path_ET = '/imaging/hauk/users/fm02/MEG_NEOS/ET_data'

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
    5 : ('meg22_193', '221007'),
    6 : ('meg22_194', '221010'),   
#    7 : ('meg22_195', '221011'), # participant did not get MRI
    8 : ('meg22_196', '221011'),
    9 : ('meg22_197', '221011'), 
    10 : ('meg22_198', '221012'),
    11 : ('meg22_199', '221014'),
    12 : ('meg22_202', '221019'),
    13 : ('meg22_203', '221020'),
    14 : ('meg22_204', '221020'),
    15 : ('meg22_206', '221021'),
    16 : ('meg22_207', '221024'),
    17 : ('meg22_209', '221031'),
    18 : ('meg22_210', '221101'),
    19 : ('meg22_213', '221103'),
#20 : TOO MAGNETIC DID NOT TEST    
    21 : ('mwg22_226', '221116'), # careful, misspelled
    22 : ('meg22_228', '221117'),
    23 : ('meg22_229', '221118'),
    24 : ('meg22_232', '221122'),
    25 : ('meg22_235', '221124'),
    26 : ('meg22_245', '221208'),
    27 : ('meg22_246', '221209'),
    28 : ('meg23_025', '230216'),
    29 : ('meg23_031', '230221'),
    30 : ('meg23_034', '230222')
}

# which files to maxfilter and how to name them after sss
# [before maxfilter], [after maxfilter], [condition labels],
# [presentation/oddball frequencies]

sss_map_fnames = {
    # 0: (['pilot00_raw', 'pilot01_raw'],
    #     ['pilot00_sss_raw', 'pilot01_sss_raw']),
    # 1: (['trigger_test_raw'],
    #     ['trichk_sss_raw'])
    0 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    1 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    2 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    3 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']), 
#   4 : ([],[]),  
    5 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    6 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    7 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),       
    8 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),       
    9 :  (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),   
    10 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),
    11 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),  
    12 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),  
    13 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),  
    14 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),  
    15 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),     
    16 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    17 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    18 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),                    
    19 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
#   20 : ([],[]),    
    21 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    22 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    23 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    24 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']), 
    25 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    26 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    27 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    28 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']), 
    29 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),    
    30 : (['block1_raw', 'block2_raw', 'block3_raw', 'block4_raw', 'block5_raw'],
        ['block1_sss_raw', 'block2_sss_raw', 'block3_sss_raw', 'block4_sss_raw', 'block5_sss_raw']),       
}       



###############################################################################
# Bad channels

# include channels that stand out in empirical covariance computation
# this is usually the electrode the closest to the right eye, so
# we decided to drop that channel and the homologue for source estimation

bad_channels_all = {
    1 : {'eeg': ['EEG004', 'EEG008'],
         'meg': []},
    2 : {'eeg': ['EEG004', 'EEG008', 'EEG017'],
        'meg': []},
    3 : {'eeg': ['EEG004', 'EEG008', 'EEG029'], #check if need to add 4 (would prefer not as close to eyes)
         'meg': []},
    # 4 : {'eeg': [],
    #      'meg': []},
    5 : {'eeg': ['EEG004', 'EEG008', 'EEG017'],
        'meg': []},
    6 : {'eeg': ['EEG004', 'EEG008', 'EEG002', 'EEG029', 'EEG039'],
         'meg': []},
    7 : {'eeg': ['EEG004', 'EEG008', 'EEG054'],
         'meg': []},
    8 : {'eeg': ['EEG004', 'EEG008', 'EEG034'],
        'meg': []},
    9 : {'eeg': ['EEG004', 'EEG008'],
         'meg': []},
    10 : {'eeg': ['EEG004', 'EEG008'],
         'meg': []},
    11 : {'eeg': ['EEG004', 'EEG008', 'EEG037'],
        'meg': []},
    12 : {'eeg': ['EEG004', 'EEG008', 'EEG003', 'EEG045'], #check if need to add 8 (would prefer not as close to eyes)
         'meg': []},         
    13 : {'eeg': ['EEG004', 'EEG008', 'EEG029', 'EEG034', 'EEG061'],
         'meg': []},
    14 : {'eeg': ['EEG004', 'EEG008'],
        'meg': []},
    15 : {'eeg': ['EEG004', 'EEG008', 'EEG061'],
         'meg': []},
    16 : {'eeg': ['EEG004', 'EEG008', 'EEG002'],
          'meg': []},
    17 : {'eeg': ['EEG004', 'EEG008', 'EEG018', 'EEG039', 'EEG061'],
        'meg': []},
    18 : {'eeg': ['EEG004', 'EEG008', 'EEG045'],
         'meg': []},
    19 : {'eeg': ['EEG004', 'EEG008', 'EEG002', 'EEG063', 'EEG034'],
         'meg': []},
    # 20 : {'eeg': [],
    #     'meg': []},
    21 : {'eeg': ['EEG004', 'EEG008', 'EEG028', 'EEG029', 'EEG030', 'EEG040', 'EEG018'],
         'meg': []},
    22 : {'eeg': ['EEG004', 'EEG008', 'EEG040'],
         'meg': []},
    23 : {'eeg': ['EEG004', 'EEG008'],
        'meg': []},
    24 : {'eeg': ['EEG004', 'EEG008', 'EEG041', 'EEG050'],
         'meg': []},    
    25 : {'eeg': ['EEG004', 'EEG008', 'EEG040', 'EEG047'],
          'meg': []},
    26 : {'eeg': ['EEG004', 'EEG008', 'EEG028', 'EEG054', 'EEG002'],
          'meg': []}, 
    27 : {'eeg': ['EEG004', 'EEG008'],
         'meg': []},
    28 : {'eeg': ['EEG004', 'EEG008', 'EEG010', 'EEG029'], # participants bad channels are plenty (22,*29*,33,43,44,45,*63*) # the problem is that they are not always bad
         'meg': []},  
    29 : {'eeg': ['EEG004', 'EEG008', 'EEG034', 'EEG035', 'EEG045', 'EEG050'], # check if want to add also 50
         'meg': []},         
    30 : {'eeg': ['EEG004', 'EEG008', ],
         'meg': []},                
}

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
    4 : '',
    5 : '',
    6 : '',
    7 : '',
    8 : '',
    9 : '',
    10 : '',
    11 : '',
    12 : '',
    13 : '',
    14 : '',
    15 : '',
    16 : '',
    17 : '',
    18 : '',
    19 : '',
    20 : '',
    21 : '',
    22 : '',
    23 : '',
    24 : '',
    25 : '',
    26 : '',
    27 : '',
    28 : '',
    29 : '',
    30 : ''
}

# Artefact rejection thresholds
# for ICA, covariance matrix
reject = dict(grad=4e-10, mag=1e-11, eeg=1e-3)

###############################################################################
# ERPs

# artefact rejection thresholds for epoching
epo_reject = dict(grad=3000e-13,
                  mag=3500e-15,
                  eeg=200e-6)

epo_flat = dict(grad=1e-13,
                mag=1e-15,
                eeg=1e-6)


#####
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
l_freq, h_freq = 0.5, 40.

raw_ICA_suff = 'ica_raw'

# EDF Label start trial
edf_start_trial = 'TRIGGER 94'
# EDF Label end trial
edf_end_trial = 'TRIGGER 95'

# The values below are the triggers value that will be inserted in the event strucure.
# This allows to have all eye events in the raw data.

# Saccade triggers value 
sac_trig_value = 801 # start

# Fixation triggers value
fix_trig_value = 901 # start

# Blink triggers value
blk_trig_value = 701 # start

########################################################
# Edited for FPVS up to here
########################################################

### Epoching, Averaging

# stimulus projector delay
delay = 0.0345

# Source Space
stc_morph = 'fsaverage'

# vertex size
src_spacing = 5

subjects_dir = '/imaging/hauk/users/fm02/MEG_NEOS/MRI'

ovr_procedure = {1: 'nover',
                 2: 'ovrw',
                 3: 'ovrw',
                 5: 'ovrw',
                 6: 'ovrw',
                 8: 'ovrw',
                 9: 'ovrw',
                 10: 'ovrwonset',
                 11: 'ovrwonset',
                 12: 'ovrwonset',
                 13: 'ovrw',
                 14: 'nover',
                 15: 'ovrw',
                 16: 'ovrw',
                 17: 'ovrw',
                 18: 'ovrw',
                 19: 'ovrw',
                 21: 'ovrwonset',
                 22: 'ovrwonset',
                 23: 'ovrwonset',
                 24: 'ovrw',
                 25: 'ovrw',
                 26: 'ovrw',
                 27: 'ovrw',
                 28: 'ovrw',
                 29: 'ovrw',
                 30: 'ovrw'}

# ovr_procedure_old = {1: 'ovrons',
#                   2: 'ovr',
#                   3: 'ovr',
#                   5: 'ovrons',
#                   6: 'novr',
#                   8: 'novr',
#                   9: 'ovrons',
#                   10: 'novr',
#                   11: 'ovrons',
#                   12: 'ovr',
#                   13: 'ovr',
#                   14: 'ovr',
#                   15: 'ovrons',
#                   16: 'novr',
#                   17: 'ovrons',
#                   18: 'novr',
#                   19: 'novr',
#                   21: 'ovrons',
#                   22: 'novr',
#                   23: 'ovrons',
#                   24: 'ovrons',
#                   25: 'ovrons',
#                   26: 'ovrons',
#                   27: 'ovrons',
#                   28: 'ovr',
#                   29: 'ovr',
#                   30: 'ovrons'
# }


# ovr_procedure_old = {1: 'ovrons',
#                  2: 'novr',
#                  3: 'ovr',
#                  5: 'ovrons',
#                  6: 'ovr',
#                  8: 'novr',
#                  9: 'ovrons',
#                  10: 'novr',
#                  11: 'ovrons',
#                  12: 'ovr',
#                  13: 'ovr',
#                  14: 'ovr',
#                  15: 'ovrons',
#                  16: 'novr',
#                  17: 'ovrons',
#                  18: 'novr',
#                  19: 'novr',
#                  21: 'ovrons',
#                  22: 'novr',
#                  23: 'ovrons',
#                  24: 'ovrons',
#                  25: 'ovrons',
#                  26: 'ovrons',
#                  27: 'ovrons',
#                  28: 'ovr',
#                  29: 'ovr',
#                  30: 'ovrons'}
