#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=PGHash     # sets the job name if not set from environment
#SBATCH --time=20:30:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=furongh    # set QOS, this will determine what resources can be requested
#SBATCH --qos=high    # set QOS, this will determine what resources can be requested
#SBATCH --partition=dpart
#SBATCH --gres=gpu:4
#SBATCH --ntasks=10
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load mpi
module load cuda/11.4.4
module load cudnn/v8.2.1

mpirun -np 2 python run_cluster.py --mca btl_openib_warn_no_device_params_found 0 --mca orte_base_help_aggregate 0 --mca btl ^openib ...
