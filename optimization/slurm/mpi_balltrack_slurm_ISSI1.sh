#!/bin/sh

## Job name to distinguish it in the queue
#SBATCH --job-name=MPIballtrack_ISSI1
#SBATCH --partition=normal

## Separate output and error messages into 2 files
#SBATCH --output=/scratch/%u/%x-%N-%j.out
#SBATCH --error=/scratch/%u/%x-%N-%j.err
## Task count reduced for better scheduling and node alignment
#SBATCH --ntasks=128

## Memory per CPU core (Reduced to 2G for Case B to drastically reduce SLURM queue wait times)
#SBATCH --mem-per-cpu=4G

## Increased walltime to account for fewer parallel workers
#SBATCH --time=1-00:00:00

## Notification settings
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rattie@gmu.edu

umask 0027

set -e
set -x

## Load the relevant modules needed for the job
## We rely on the cluster's default module hierarchy to automatically activate UCX.
module load gnu10
module load python
module load openmpi4/4.1.2
source ~/envs/MPIpool/bin/activate

## Export project-specific paths
export DATA=/scratch/rattie/Data
export PYTHONPATH=~/dev/sunflower
export MAX_CPUS=$SLURM_NTASKS

## Run using the $SLURM_NTASKS parallel workers automatically defined by the --ntasks option
## Use ob1 PML with TCP/shared-memory BTLs, excluding lo, docker0, and virbr0 to ensure stable inter-node routing and bypass UCX errors
mpirun --mca pml ob1 --mca btl self,vader,tcp --mca btl_tcp_if_exclude lo,docker0,virbr0 -np $SLURM_NTASKS python -m mpi4py.futures ~/dev/sunflower/optimization/balltrack_parameter_sweep_ISSI1.py

# Run aggregation script after balltracking completes
python ~/dev/sunflower/optimization/parameter_sweep_aggregation_ISSI1.py

# Navigate to the output directory defined in inputs.py
# cd "$DATA/sanity_check/stein_series/calibration4"
# # Create a compressed tar archive of the output subfolders using zstd
# # -T0 tells zstd to use all available CPU cores
# tar -cvf - mean_velocity_files param_sweep_files | zstd -T0 > optimization_results.tar.zst