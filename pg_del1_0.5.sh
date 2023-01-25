#!/usr/bin/env bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=p1d0.25    # sets the job name if not set from environment
#SBATCH --time=24:00:00    # how long you think your job will take to complete; format=hh:mm:ss
#SBATCH --account=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --qos=scavenger    # set QOS, this will determine what resources can be requested
#SBATCH --partition=scavenger
#SBATCH --ntasks=1
#SBATCH --mem 128gb         # memory required by job; if unit is not specified MB will be assumed
#SBATCH --nice=0
#SBATCH --mail-type=END   # Valid type values are NONE, BEGIN, END, FAIL, REQUEUE
#SBATCH --nodelist=cml09

module purge
module load mpi
module load cuda/11.4.4
source ../../../../cmlscratch/marcob/environments/pghash/bin/activate

mpirun -n 1 python run_pg.py --hash_type pghash --steps_per_test 100 --train_bs 128 --dataset Delicious200K --cr 0.5 --q 1 --name exp-t1 --randomSeed 1203