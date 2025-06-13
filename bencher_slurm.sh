#!/bin/bash
#SBATCH --job-name=vlsat3-z3-benchmark
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=compute-p1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:50:00
#SBATCH --output=logs/vlsat3-z3-%j.out
#SBATCH --error=logs/vlsat3-z3-%j.err

set -e

# 1) load & activate conda
module load miniconda3
conda activate /scratch/vmitev/Benchmarking/my-conda-env

# 2) benchExec clone on PYTHONPATH
export PYTHONPATH="$HOME/benchExec-dev:$PYTHONPATH"

# 3) workspace
cd /scratch/vmitev/Benchmarking

python3 bencher.py
