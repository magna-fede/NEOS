#!/bin/bash
#SBATCH --job-name=Unfolding  # Name this job
#SBATCH --output=slurm_%u_%x_%j_stdout.log          # Name of log for STDOUT & STDERR
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --verbose                                   # Be verbose wherever possible
#SBATCH --time=48:00:00                             # Request resources for 24 hours
#SBATCH --mail-type=end,fail                        # Email on job completion / failure


# Set up environment
WORKDIR="/home/fm02/MEG_NEOS/NEOS"

# Path to script
SCRIPT="temp_UnfoldOverlapCorrection_allfixations.ipy"

# Make folders for logging
LOGDIR="/home/fm02/Desktop/MEG_EOS_scripts/sbatch_out"
mkdir -p "$LOGDIR/tasks"

# Activate conda environment (or some other module that manages environment)
conda activate mnejl

# csh -c 'source ~/.cshrc; exec bash'

echo "JOB $SLURM_JOB_ID STARTING"

# Loop over range of arguments to script
# array=(1 12)
# for i in "${array[@]}"
# for i in {1..30}
# do
echo "TASK $i STARTING"

# Run task on node
srun --ntasks=1 \
    --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
    --exclusive "ipython" $SCRIPT $i &

echo "TASK $i PUSHED TO BACKGROUND"
# done

# # Wait till everything has run
wait

# # # for i in "${array[@]}"
# for i in {1..30}
# do
#     echo "TASK $i STARTING"

#     # Run task on node
#     srun --ntasks=1 \
#         --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
#         --exclusive "python" $SCRIPT2 $i &
    
#     echo "TASK $i PUSHED TO BACKGROUND"
# done

# # Wait till everything has run
# wait

# # for i in "${array[@]}"
# for i in {1..30}
# do
#     echo "TASK $i STARTING"

#     # Run task on node
#     srun --ntasks=1 \
#         --output="$LOGDIR/tasks/slurm_%u_%x_%A_%a_%N_stdout_task_$i.log" \
#         --exclusive "python" $SCRIPT3 $i &
    
#     echo "TASK $i PUSHED TO BACKGROUND"
# done

# # Wait till everything has run
# wait

echo "JOB $SLURM_JOB_ID COMPLETED"



