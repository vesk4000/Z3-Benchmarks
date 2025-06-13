#!/bin/bash
#SBATCH --job-name=vlsat-family-a
#SBATCH --partition=compute-p1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2800MB
#SBATCH --time=03:55:00
#SBATCH --output=../logs/%x-%j.out
#SBATCH --error=../logs/%x-%j.err
#SBATCH --mail-type=BEGIN
#SBATCH --account=education-eemcs-courses-cse3000

module load miniconda3
conda activate /scratch/vmitev/Benchmarking/my-conda-env
cd /scratch/vmitev/Benchmarking

python3 benchy.py \
	--name vlsat-family-a \
	--time-limit 00:40:00 \
	--memory-limit 2500MB \
	--glob "**/vlsat3_a*.smt2" \
	--threads 32
