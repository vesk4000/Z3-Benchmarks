#!/bin/bash
#SBATCH --job-name=vlsat3_a
#SBATCH --partition=compute-p1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3200MB
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
conda activate new-conda-env
cd /scratch/vmitev/Benchmarking

python3 benchy.py --name vlsat3_a
