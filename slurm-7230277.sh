#!/bin/bash
#SBATCH --job-name=vlsat3-z3-benchmark
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:50:00
#SBATCH --output=vlsat3-z3-%j.out
#SBATCH --error=vlsat3-z3-%j.err

set -e

# 1) load & activate conda
module load miniconda3
conda activate /scratch/vmitev/Benchmarking/my-conda-env

# 2) benchExec clone on PYTHONPATH
export PYTHONPATH="$HOME/benchExec-dev:$PYTHONPATH"

# 3) workspace
cd /scratch/vmitev/Benchmarking

# 5) run BenchExec SLURM‐wrapper instead of direct apptainer exec
python3 ~/benchExec-dev/contrib/slurm-benchmark.py \
    --slurm \
    --no-container \
    --no-hyperthreading \
    -d \
    --scratchdir ./tmp \
    -N 16 \
    -c 1 \
    vlsat3-benchmark2.xml
