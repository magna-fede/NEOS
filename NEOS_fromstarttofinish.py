#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This collection gets you from Maxfiltered raw data to source timecourses.

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

# # This script takes care of fixing electrodes position, possibly redundant, but no harm in running it

# from NEOS_fix_electrodes import run_fix_electrodes

# # This script takes care of filtering and saving the filtered data

from NEOS_filter_raw import run_filter_raw
# # These scripts synch the MEG and ET data, either per block or concatenating each block

# from NEOS_synch_per_block import synchronise as synchronise_per_block
# from NEOS_synchronisation_includeEDF import synchronise_concat 

# # Apply best ICA for each participant and create evoked for each condition

# from NEOS_applyICA_evoked import create_evoked

# # PLot various ICA procedures and save plots
from plot_conditionoverweighting_procedure import plot_evoked_for_comparisons
# #  THIS will run through all the scripts above, careful is going to take time

if len(sys.argv) == 1:

    sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
                21,22,23,24,25,26,27,28,29,30]

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:
    # run_fix_electrodes(ss)
    # run_filter_raw(ss)    
    # synchronise_per_block(ss)
    # synchronise_concat(ss)
    # create_evoked(ss)
    plot_evoked_for_comparisons(ss)
    
    
    
# # %% SOURCE SPACE PREPARATION

# from NEOS_makeSourceSpace import make_source_space
# from NEOS_makeBem import  run_make_bem
# from NEOS_MakeForwardModel import run_make_forward_solution
# from NEOS_ComputeCovariance import compute_covariance
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
# #     run_make_forward_solution(ss)
#     compute_covariance(ss, cov_method='empirical', save_covmat=True, plot_covmat=True)
# #     make_InverseOperator(ss)
    
# %% SOURCE SPACE ANALYSIS

# from NEOS_stcsFactorialDesign import compute_stcs
# from NEOS_MorphStcsFsaverage import compute_morphed_stcs
# from NEOS_stcsFactorialDesign_forFtest import compute_stcs as compute_stcs_forF
# from NEOS_MakeStcsROIs_SingleEpochs import make_stcsEpochs

# from NEOS_permutation_Ttest import 
# from NEOS_permutationFtest import 


# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     # compute_stcs(ss, method='eLORETA')
#     make_stcsEpochs(ss, method='eLORETA')
# #    compute_morphed_stcs(ss, stc_sub='eLORETA')    
    
# %% ADDITIONAL SCRIPTS

# from NEOS_getbetas_forsource import get_betas

# if len(sys.argv) == 1:

#     sbj_ids = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,
#                21,22,23,24,25,26,27,28,29,30]

# else:

#     # get list of subjects IDs to process
#     sbj_ids = [int(aa) for aa in sys.argv[1:]]


# for ss in sbj_ids:
#     get_betas(ss)