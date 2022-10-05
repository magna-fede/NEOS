#!/imaging/local/software/miniconda/envs/mne0.20/bin/python
"""
==========================================
Try to submit sbatch jobs for Eye On Semantics
SLURM, Python 3
==========================================

fm02
"""

import subprocess
from os import path

from importlib import reload

# import study parameters
import EOS_config as config
reload(config)

print(__doc__)

# wrapper to run python script via qsub. Python3
fname_wrap = path.join('/', 'home', 'fm02', 'Desktop', 'MEG_EOS_scripts',
                     'Python2SLURM.sh')

# indices of subjects to process
subjs = config.do_subjs

job_list = [
    ### Neuromag Maxfilter
    {'N':   'F_MF',                  # job name
     'Py':  'EOS_Maxfilter',  # Python script
     'Ss':  subjs,                    # subject indices
     'mem': '16G',                   # memory for qsub process
     'dep': '',                       # name of preceeding process (optional)
     'node': '--constraint=maxfilter'},  # node constraint for MF, just picked one

    # ### fix EEG electrode positions in fiff-files
    # ### NOTE: Can get "Permission denied"; should be run separately
    # {'N':   'F_FE',                    # job name
    #  'Py':  'EOS_fix_electrodes',      # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '1G',                    # memory for qsub process
    #  'dep': ''},                       # name of preceeding process (optional)

    # ### Pre-processing

    # ### Filter raw data
    # {'N':   'F_FR',                  # job name
    #  'Py':  'EOS_filter_raw',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '16G',                    # memory for qsub process
    #  'dep': ''},                      # name of preceeding process (optional)

    # {'N':   'F_Cov',                  # job name
    #  'Py':  'FPVS_make_covmat',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '4G',                    # memory for qsub process
    #  'dep': 'F_FR'},

    #  ### Compute ICA
    # {'N':   'F_CICA',                  # job name
    #  'Py':  'EOS_Compute_ICA',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '96G',                    # memory for qsub process
    #  'dep': ''},                      # name of preceeding process (optional)
      ### Apply ICA (change ica_suff in config_sweep.py if necessary)
    # {'N':   'F_AICA',                  # job name
    #  'Py':  'EOS_Apply_ICA',          # Python script
    #  'Ss':  subjs,                    # subject indices
    #  'mem': '2G',                    # memory for qsub process
    #  'dep': ''},                      # name of preceeding process (optional)

]

# directory where python scripts are
dir_py = path.join('/', 'home', 'fm02', 'Desktop', 'MEG_EOS_scripts')

# directory for qsub output
dir_sbatch = path.join('/', 'home', 'fm02', 'Desktop', 'MEG_EOS_scripts',
                     'sbatch_out')

# keep track of qsub Job IDs
Job_IDs = {}

for job in job_list:

    for Ss in job['Ss']:

        Ss = str(Ss)  # turn into string for filenames etc.

        N = Ss + job['N']  # add number to front
        Py = path.join(dir_py, job['Py'])
        mem = job['mem']

        # files for qsub output
        file_out = path.join(dir_sbatch,
                           job['N'] + '_' + '-%s.out' % str(Ss))
        file_err = path.join(dir_sbatch,
                           job['N'] + '_' + '-%s.err' % str(Ss))

        # if job dependent of previous job, get Job ID and produce command
        if 'dep' in job:  # if dependency on previous job specified
            if job['dep'] == '':
                dep_str = ''
            else:
                job_id = Job_IDs[Ss + job['dep'], Ss]
                dep_str = '--dependency=afterok:%s' % (job_id)
        else:
            dep_str = ''

        if 'node' in job:  # if node constraint present (e.g. Maxfilter)
            node_str = job['node']
        else:
            node_str = ''

        # if 'var' in job:  # if variables for python script specified
        #     var_str = job['var']
        # else:
        #     var_str = ''

        # sbatch command string to be executed
        sbatch_cmd = 'sbatch \
                        -o %s \
                        -e %s \
                        --export=pycmd="%s.py",subj_idx=%s, \
                        --mem=%s -t 1-00:00:00 %s -J %s %s %s' \
                        % (file_out, file_err, Py, Ss, mem,
                           node_str, N, dep_str, fname_wrap)

        # format string for display
        print_str = sbatch_cmd.replace(' ' * 25, '  ')
        print('\n%s\n' % print_str)

        # execute qsub command
        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, shell=True)

        # get linux output
        (out, err) = proc.communicate()

        # keep track of Job IDs from sbatch, for dependencies
        Job_IDs[N, Ss] = str(int(out.split()[-1]))
