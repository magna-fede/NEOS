#!/bin/bash
#SBATCH --job-name=permtest1  # Name this job
#SBATCH --output=slurm_%u_%x_%j_stdout.log          # Name of log for STDOUT & STDERR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8G
#SBATCH --verbose                                   # Be verbose wherever possible
#SBATCH --time=48:00:00                             # Request resources for 24 hours
#SBATCH --mail-type=end,fail                        # Email on job completion / failure

# Set up environment
WORKDIR="/home/fm02/MEG_NEOS/NEOS"

# Path to script
# SCRIPT="$WORKDIR/NEOS_eyeCA.py"
# SCRIPT="$WORKDIR/main_over_participants.py"
# SCRIPT="$WORKDIR/apply_ica_over_participants.py"
# SCRIPT="plot_frps_over_participants.py"
# SCRIPT="$WORKDIR/NEOS_synch_per_block.py"
# SCRIPT="compare_overweighting_ica_over_participants.py"
# SCRIPT2="compare_NOoverweighting_ica_over_participants.py"
# SCRIPT="plt_frps_fileffects_icaboth_over_participants.py"
# SCRIPT="plt_frps_ic_n_filt.py"
# SCRIPT="snr_compare_componentselection_ica_oveweighted_withplots.py"
# SCRIPT2="snr_compare_componentselection_ica_NOoveweight_withplots.py"
# SCRIPT="NEOS_permutationFtest.py"
# SCRIPT="GA_unfoldeffects_sensors.py"
SCRIPT="erp_permtest.py"
# SCRIPT="wholebrain_evoked_concreteness.py"

# Make folders for logging
LOGDIR="/home/fm02/Desktop/MEG_EOS_scripts/sbatch_out"
mkdir -p "$LOGDIR/tasks"

# Activate conda environment (or some other module that manages environment)
conda activate mne1.2.1_0

echo "JOB $SLURM_JOB_ID STARTING"

# Loop over range of arguments to script
# array=(1 2 3 5 6 8 9 10 11 12 13 14 15 16 17 18 19 
#            21 22 23 24 25 26 27 28 29 30)
# for i in "${array[@]}"
# # for i in {1..30}
# do
    echo "TASK $i STARTING"

    # Run task on node
    srun --ntasks=1 \
        --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
        --exclusive "python" $SCRIPT $i &
    
    echo "TASK $i PUSHED TO BACKGROUND"
# done

# # Wait till everything has run
wait

echo "JOB $SLURM_JOB_ID COMPLETED"



