#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
SBATCH --job-name=MPIballtrack
SBATCH --partition=normal
## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file
## Slurm can send you updates via email
SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
SBATCH --mail-user=rattie@gmu.edu     # Put your GMU email address here
## Specify how much memory your job needs. (2G is the default)
SBATCH --mem=66G        # Total memory needed per task (units: K,M,G,T)
## Specify how much time your job needs. (default: see partition above)
SBATCH --time=0-02:00   # Total time needed for job: Days-Hours:Minutes
SBATCH --ntasks=33   # 50 workers, 1 manager
set echo
umask 0027

## Load the relevant modules needed for the job
module load gnu10
module load python
module load openmpi4/4.1.2
source ~/envs/MPIpool/bin/activate
export DATA3=/groups/RATTIE/Data
export PYTHONPATH=~/dev/sunflower
