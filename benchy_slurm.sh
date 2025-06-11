#!/bin/bash
#SBATCH --job-name=benchy_slurm
#SBATCH --mail-type=BEGIN
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=compute-p1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2800MB
#SBATCH --time=03:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e

# 1) load & activate conda
module load miniconda3
conda activate /scratch/vmitev/Benchmarking/my-conda-env

# 2) benchExec clone on PYTHONPATH
export PYTHONPATH="$HOME/benchExec-dev:$PYTHONPATH"

# 3) workspace
cd /scratch/vmitev/Benchmarking

python3 benchy.py
