#!/bin/bash
#SBATCH --job-name=unfold  # Name this job
#SBATCH --output=slurm_%u_%x_%j_stdout.log          # Name of log for STDOUT & STDERR
#SBATCH --array=1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --verbose                                   # Be verbose wherever possible
#SBATCH --time=72:00:00                             # Request resources for 24 hours
#SBATCH --mail-type=end,fail                        # Email on job completion / failure

# Set up environment
WORKDIR="/home/fm02/MEG_NEOS/NEOS"



SCRIPT1="$WORKDIR/unfold_eeg.jl"
SCRIPT2="$WORKDIR/unfold_meg.jl"

# Make folders for logging
LOGDIR="/home/fm02/Desktop/MEG_EOS_scripts/sbatch_out"
mkdir -p "$LOGDIR/tasks"

# # Activate conda environment (or some other module that manages environment)
# conda activate mne1.2.1_0

echo "JOB $SLURM_JOB_ID STARTING"

# Run task on node
srun --ntasks=1 \
    --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_ $SLURM_ARRAY_TASK_ID.log" \
    julia $SCRIPT1 $SLURM_ARRAY_TASK_ID &


# # Wait till everything has run
wait


# Run task on node
srun --ntasks=1 \
    --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_ $SLURM_ARRAY_TASK_ID.log" \
    julia $SCRIPT2 $SLURM_ARRAY_TASK_ID &


# # Wait till everything has run
wait


echo "JOB $SLURM_JOB_ID COMPLETED"



