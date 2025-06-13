#!/bin/bash
#SBATCH --job-name=vlsat3-z3-benchmark
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --partition=compute-p1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:22:00
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

srun --quit-on-interrupt -t 0:20:0 -c 1 --mem 3000M --threads-per-core=1 --ntasks=1 sh -c '/home/vmitev/bin/z3 -smt2 tactic.default_tactic="(then simplify propagate-values solve-eqs ctx-simplify simplify smt)" datasets/VLSAT3/cadp.inria.fr/ftp/benchmarks/vlsat/vlsat3_g00.smt2'

srun --quit-on-interrupt -t 0:20:0 -c 1 --mem 3000M --threads-per-core=1 --ntasks=1 sh -c '/home/vmitev/bin/z3 -smt2 tactic.default_tactic="(then simplify propagate-values solve-eqs ctx-simplify simplify smt)" datasets/VLSAT3/cadp.inria.fr/ftp/benchmarks/vlsat/vlsat3_g01.smt2'

srun --quit-on-interrupt -t 0:20:0 -c 1 --mem 3000M --threads-per-core=1 --ntasks=1 sh -c '/home/vmitev/bin/z3 -smt2 tactic.default_tactic="(then simplify propagate-values solve-eqs ctx-simplify simplify smt)" datasets/VLSAT3/cadp.inria.fr/ftp/benchmarks/vlsat/vlsat3_g02.smt2'

srun --quit-on-interrupt -t 0:20:0 -c 1 --mem 3000M --threads-per-core=1 --ntasks=1 sh -c '/home/vmitev/bin/z3 -smt2 tactic.default_tactic="(then simplify propagate-values solve-eqs ctx-simplify simplify smt)" datasets/VLSAT3/cadp.inria.fr/ftp/benchmarks/vlsat/vlsat3_g03.smt2'
