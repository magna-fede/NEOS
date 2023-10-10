#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collection gets you from Maxfiltered raw data to source timecourses and beyond.

NB: This ignores the steps for calculatin the most optimal ICA procedure for each subject.
That will need to be run and inferred separately.

@author: federica.magnabosco@mrc-cbu.cam.ac.uk
"""

import sys
import os
from os import path
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

import mne

os.chdir("/home/fm02/MEG_NEOS/NEOS")
import NEOS_config as config

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)


# #%% PREPROCESSING

# MF command on TRIUX machine
# from NEOS_Maxfilter import run_maxfilter

# # This script takes care of fixing electrodes position, possibly redundant, but no harm in running it

# from NEOS_fix_electrodes import run_fix_electrodes

# # This script takes care of filtering and saving the filtered data

# from NEOS_filter_raw import run_filter_raw
# # These scripts synch the MEG and ET data, either per block or concatenating each block

# from NEOS_synch_per_block import synchronise as synchronise_per_block
# from NEOS_synchronisation_includeEDF import synchronise_concat 

# # Apply best ICA for each participant and create evoked for each condition

# from NEOS_applyICA_evoked_dropbadchannels import create_evoked_from_raw # old, do not use
# from NEOS_applyICA_evoked_dropbadchannels import create_evoked_from_ICA_raw as create_evoked # for source estimation
# from NEOS_applyICA_evoked_dropbadchannels import create_evoked_from_ICA_raw_keepallchannels as create_evoked # for sensor space sanalysis
# # PLot various ICA procedures and save plots
# from plot_conditionoverweighting_procedure import plot_evoked_for_comparisons
# #  THIS will run through all the scripts above, careful is going to take time

# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                 21,22,23,24,25,26,27,28,29,30]

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
    # run_maxfilter(ss)
    # run_fix_electrodes(ss)
    # run_filter_raw(ss)    
    # synchronise_per_block(ss)
    # synchronise_concat(ss)
    # create_evoked(ss)
#     plot_evoked_for_comparisons(ss)
    
    
    
# # %% SOURCE SPACE PREPARATION

# from NEOS_makeSourceSpace import make_source_space
# from NEOS_makeBem import  run_make_bem
# from NEOS_MakeForwardModel import run_make_forward_solution
# from NEOS_ComputeCovariance import compute_covariance
# from NEOS_ComputeCovariance_dropbadchannels import compute_covariance_MEGonly_from_ICA_raw as compute_covariance
# from NEOS_MakeInverseOperator import make_InverseOperator

# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                 21,22,23,24,25,26,27,28,29,30]

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
# #     make_source_space(ss)
# #     run_make_bem(ss)    
#     # run_make_forward_solution(ss)
#     # compute_covariance(ss, cov_method=['auto'], save_covmat=True, plot_covmat=False)
#     if ss==12:
#         make_InverseOperator(ss, fwd='MEG', MEGonly=True, cov='MEGonly_auto_dropbads', inv_suf='MEGonly_auto_dropbads')
#     else:
#         make_InverseOperator(ss, MEGonly=True, cov='MEGonly_auto_dropbads', inv_suf='MEGonly_auto_dropbads')
    
# %% SOURCE SPACE ANALYSIS

# from generate_fsaverage_SNlabels import create_fsaverage_rois

# create_fsaverage_rois()

from NEOS_stcsFactorialDesign import compute_evoked_condition_stcs as compute_stcs
from NEOS_stcsFactorialDesign import compute_unfold_evoked_condition_stcs as compute_unfold_stcs
from NEOS_MorphStcsFsaverage import compute_morphed_stcs
from NEOS_MorphStcsFsaverage import compute_unfold_morphed_stcs
from NEOS_stcsFactorialDesign import stcs_inlabel_from_stc
from NEOS_stcsFactorialDesign import stcs_unfold_inlabel_from_stc



if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
                21,22,23,24,25,26,27,28,29,30]

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    compute_stcs(ss, MEGonly=True, method='eLORETA', inv_suf='MEGonly_auto_dropbads')
    compute_unfold_stcs(ss, MEGonly=True, method='eLORETA', inv_suf='MEGonly_auto_dropbads')
    compute_morphed_stcs(ss, stc_sub='eLORETA_MEGonly_auto_dropbads')   
    compute_unfold_morphed_stcs(ss, stc_sub='eLORETA_MEGonly_auto_dropbads')   
    stcs_inlabel_from_stc(ss, method='eLORETA', inv_suf='MEGonly_auto_dropbads', mode_avg='mean')
    stcs_unfold_inlabel_from_stc(ss, method='eLORETA', inv_suf='MEGonly_auto_dropbads', mode_avg='mean')

     
    
# %% ADDITIONAL SCRIPTS

# from NEOS_getbetas_forsource import get_betas
# from NEOS_applyICA_evoked import create_evoked
# from NEOS_MakeStcsROIs_SingleEpochs import make_stcsEpochs
# from NEOS_MakeStcsROIs_SingleEpochs import make_stcsEpochs_intensities
# from NEOS_MakeStcsROIs_SingleEpochs import make_stcsEpochs_factorial
# from NEOS_MakeStcsROIs_SingleEpochs_decoding import get_decoding_avg3trials_scores
# from NEOS_MakeStcsROIs_SingleEpochs_decoding import decoding_continuous_predictors
# from NEOS_decoding_ConcPred_sensor import get_decoding_sensor_avg3trials_scores

# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                 21,22,23,24,25,26,27,28,29,30]

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     create_evoked(ss)
    # get_betas(ss)
    # make_stcsEpochs(ss)
    # get_decoding_sensor_avg3trials_scores(ss)
    # get_decoding_avg3trials_scores(ss, inv_suf='auto_dropbads')
    # decoding_continuous_predictors(ss)
    # make_stcsEpochs_intensities(ss)
    # make_stcsEpochs_factorial(ss)
    
    