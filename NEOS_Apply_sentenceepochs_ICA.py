#!/imaging/local/software/miniconda/envs/mne0.20/bin/python
"""
Apply ICA for MEG NEOS.
Decompostion computed using sentences as epochs (0-padded)

author: federica.magnabosco@mrc-cbu.cam.ac.uk 
"""

import sys

from os import path
import numpy as np

from importlib import reload

import mne

print('MNE Version: %s\n\n' % mne.__version__)  # just in case
print(mne)

import NEOS_config as config
reload(config)

###############################################
### Parameters
###############################################


# "emulate" the args from ArgParser in Fiff_Apply_ICA.py
# filenames depend on subject, the rest are variables
class CreateArgs:
    """Parser for input arguments."""

    def __init__(self, FileRawIn, FileICA, FileRawOut):
        self.FileRawIn = FileRawIn
        self.FileICA = FileICA
        self.FileRawOut = FileRawOut


def run_Apply_ICA(sbj_id):
    """Apply previously computed ICA to raw data."""

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
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf))

    print('Concatenating data')
    data = mne.concatenate_raws(data)
    
    print(f'Reading raw file {sss_map_fnames}')
    data = []
    for drf in data_raw_files:
        data.append(mne.io.read_raw_fif(drf))

    print('Concatenating data')
    data = mne.concatenate_raws(data)
    
    data.load_data()
        ###
        # APPLY ICA
        ###

 

    print('Reading ICA file')
    ica = mne.preprocessing.read_ica(path.join(sbj_path, sbj_path[-3:] + "_sss_f_raw-ica.fif"))

    print('Applying ICA to raw file')
    ica.apply(data)

    data.save(path.join(sbj_path, sbj_path[-3:] + "_sss_f_ica_sentepochs_raw.fif"), overwrite=True)

# get all input arguments except first
if len(sys.argv) == 1:

    sbj_ids = np.arange(0, len(config.map_subjects)) + 1

else:

    # get list of subjects IDs to process
    sbj_ids = [int(aa) for aa in sys.argv[1:]]


for ss in sbj_ids:

    run_Apply_ICA(ss)
