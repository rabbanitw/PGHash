#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=slide-amz4    # sets the job name if not set from environment
#SBATCH --time=70:00:00    # how long you think your job will take to complete; format=hh:mm:ss
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
source ../../../../cmlscratch/marcob/environments/pghash/bin/activate

mpirun -n 4 python run_pg.py --hash_type slide --dwta 1 --steps_per_test 100 --train_bs 256 --dataset Amazon670K --cr 1 --epochs 40 --name slide --randomSeed 239 --c 8 --sdim 8 --num_tables 50 --steps_per_lsh 50