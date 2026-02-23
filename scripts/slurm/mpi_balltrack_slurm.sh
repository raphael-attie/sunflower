#!/bin/sh

## Job name to distinguish it in the queue
#SBATCH --job-name=MPIballtrack
#SBATCH --partition=normal

## Separate output and error messages into 2 files
#SBATCH --output=/scratch/%u/%x-%N-%j.out
#SBATCH --error=/scratch/%u/%x-%N-%j.err

## Task count reduced for better scheduling and node alignment
#SBATCH --ntasks=64

## Memory per CPU core (adjust to 16G if 8G is insufficient)
#SBATCH --mem-per-cpu=8G

## Increased walltime to account for fewer parallel workers
#SBATCH --time=0-11:00

## Notification settings
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rattie@gmu.edu

set echo
umask 0027

## Load the relevant modules needed for the job
module load gnu10
module load python
module load openmpi4/4.1.2
source ~/envs/MPIpool/bin/activate

## Export project-specific paths
export DATA3=/scratch/rattie/Data
export PYTHONPATH=~/dev/sunflower

## Run using the automatically defined 64 tasks from $SLURM_NTASKS
mpirun -np $SLURM_NTASKS python -m mpi4py.futures ~/dev/sunflower/optimization/balltrack_parameter_sweep.py