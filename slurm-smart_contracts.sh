#!/bin/bash
#SBATCH --job-name=smart_contracts
#SBATCH --partition=memory
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12500MB
#SBATCH --time=03:55:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
conda activate new-conda-env
cd /scratch/vmitev/Benchmarking

python3 benchy.py --name smart_contracts
