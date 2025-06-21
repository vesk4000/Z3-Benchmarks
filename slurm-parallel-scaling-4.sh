#!/bin/bash
#SBATCH --job-name=parallel-scaling-4
#SBATCH --partition=compute-p2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3200MB
#SBATCH --time=03:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
conda activate new-conda-env
cd /scratch/vmitev/Benchmarking

python3 benchy.py --name parallel-scaling-4
