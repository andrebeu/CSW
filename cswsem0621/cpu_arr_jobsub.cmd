#!/usr/bin/env bash

#SBATCH -t 5:59:00   # runs for 48 hours (max)  
#SBATCH -N 1         # node count 
#SBATCH -c 2         # number of cores 
#SBATCH -o ./slurms/output.%j.%a.out

## scotty
# module load pyger/0.9
conda init bash
conda activate sem
## tiger
# module load anaconda3/4.4.0
# source activate sem

# get arr idx
slurm_arr_idx=${SLURM_ARRAY_TASK_ID}

# use arr idx to get params
param_str=`python get_param_jobsub.py ${slurm_arr_idx}`
echo ${param_str}

# submit job
srun python gs102121.py "${param_str}"

echo "done.sh"

# slurm diagnostics
sacct --format="CPUTime,MaxRSS"

