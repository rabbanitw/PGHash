#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=pg_vanilla2     # sets the job name if not set from environment
#SBATCH --time=10:45:00     # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE

module load mpi
module load cuda/11.4.4
module load cudnn/v8.2.1

mpirun -np 1 python run_cluster.py --dataset Delicious200K --cr 0.1 --epochs 10 --batch_size 32 --hash_type slide_avg
