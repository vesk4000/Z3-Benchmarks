#!/bin/bash
#SBATCH --job-name=slurm-parallel-hyperparameter-search
#SBATCH --partition=compute-p2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3200MB
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
conda activate new-conda-env
cd /scratch/vmitev/Benchmarking

python3 benchy.py --name parallel-hyperparameter-search
