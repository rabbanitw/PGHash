#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=test    # sets the job name if not set from environment
#SBATCH --time=02:30:00    # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --ntasks=4
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module purge
module load mpi
module load cuda/11.4.4
source ../../cmlscratch/marcob/environments/pghash/bin/activate

mpirun -np 4 python runPG.py --dataset Delicious200K --test_bs 8192

# --mca btl_openib_warn_no_device_params_found 0 --mca orte_base_help_aggregate 0 --mca btl ^openib ...
