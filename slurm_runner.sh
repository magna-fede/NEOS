#!/bin/bash
#SBATCH --job-name=compare_icas  # Name this job
#SBATCH --output=slurm_%u_%x_%j_stdout.log          # Name of log for STDOUT & STDERR
#SBATCH --ntasks=30
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --verbose                                   # Be verbose wherever possible
#SBATCH --time=72:00:00                             # Request resources for 24 hours
#SBATCH --mail-type=end,fail                        # Email on job completion / failure

# Set up environment
WORKDIR="/home/fm02/MEG_NEOS/NEOS"

# Path to script

SCRIPT3="snr_compare_componentselection_ica_oveweighted_onsets_withplots.py"
SCRIPT="snr_compare_componentselection_ica_oveweighted_withplots.py"
SCRIPT2="snr_compare_componentselection_ica_NOoveweight_withplots.py"

# SCRIPT="NEOS_rois_predictability_coherence.py"

# SCRIPT="NEOS_MorphStcsFsaverage.py"

# SCRIPT="NEOS_fromstarttofinish.py"
# SCRIPT="snr_radarplot_component_selection.py"
# SCRIPT2="snr_radarplot_filt_ovr_both.py"

# SCRIPT="NEOS_evokedasfactors.py"

# SCRIPT="NEOS_MorphStcsFsaverage.py"

# Make folders for logging
LOGDIR="/home/fm02/Desktop/MEG_EOS_scripts/sbatch_out"
mkdir -p "$LOGDIR/tasks"

# Activate conda environment (or some other module that manages environment)
conda activate mne1.2.1_0

echo "JOB $SLURM_JOB_ID STARTING"

# Loop over range of arguments to script
# array=(1 12)
# for i in "${array[@]}"
for i in {1..30}
do
    echo "TASK $i STARTING"

    # Run task on node
    srun --ntasks=1 \
        --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
        --exclusive "python" $SCRIPT $i &
    
    echo "TASK $i PUSHED TO BACKGROUND"
done

# # Wait till everything has run
wait

# for i in "${array[@]}"
for i in {1..30}
do
    echo "TASK $i STARTING"

    # Run task on node
    srun --ntasks=1 \
        --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
        --exclusive "python" $SCRIPT2 $i &
    
    echo "TASK $i PUSHED TO BACKGROUND"
done

# Wait till everything has run
wait

# for i in "${array[@]}"
for i in {1..30}
do
    echo "TASK $i STARTING"

    # Run task on node
    srun --ntasks=1 \
        --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
        --exclusive "python" $SCRIPT3 $i &
    
    echo "TASK $i PUSHED TO BACKGROUND"
done

# Wait till everything has run
wait

echo "JOB $SLURM_JOB_ID COMPLETED"



