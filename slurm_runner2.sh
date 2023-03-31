#!/bin/bash
#SBATCH --job-name=ica_testing  # Name this job
#SBATCH --output=slurm_%u_%x_%j_stdout.log          # Name of log for STDOUT & STDERR
#SBATCH --ntasks=15
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --verbose                                   # Be verbose wherever possible
#SBATCH --time=24:00:00                             # Request resources for 24 hours
#SBATCH --mail-type=end,fail                        # Email on job completion / failure

# Set up environment
WORKDIR="/home/fm02/MEG_NEOS/NEOS"

# Path to script
# SCRIPT="$WORKDIR/NEOS_eyeCA.py"
# SCRIPT="$WORKDIR/main_over_participants.py"
# SCRIPT="$WORKDIR/apply_ica_over_participants.py"
# SCRIPT="plot_frps_over_participants.py"
# SCRIPT="NEOS_synch_per_block.py"
# SCRIPT="NEOS_stcsFactorialDesign.py"
SCRIPT="NEOS_MorphStcsFsaverage_normalorientation.py"

# Make folders for logging
LOGDIR="/home/fm02/Desktop/MEG_EOS_scripts/sbatch_out"
mkdir -p "$LOGDIR/tasks"

# Activate conda environment (or some other module that manages environment)
conda activate mne1.2.1_0

echo "JOB $SLURM_JOB_ID STARTING"

# Loop over range of arguments to script
for i in {1..15}
do
    echo "TASK $i STARTING"

    # Run task on node
    srun --ntasks=1 \
        --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
        --exclusive "python" $SCRIPT $i &
    
    echo "TASK $i PUSHED TO BACKGROUND"
done

# Wait till everything has run
wait

# Loop over range of arguments to script
for i in {16..30}
do
    echo "TASK $i STARTING"

    # Run task on node
    srun --ntasks=1 \
        --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
        --exclusive "python" $SCRIPT $i &
    
    echo "TASK $i PUSHED TO BACKGROUND"
done

# Wait till everything has run
wait

echo "JOB $SLURM_JOB_ID COMPLETED"



