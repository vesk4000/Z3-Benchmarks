#!/usr/bin/env bash
set -e

# 1) load/activate conda
module load miniconda3
conda activate /scratch/vmitev/Benchmarking/my-conda-env

# 2) benchExec import
echo -n "import benchexec… "
python3 -c "import benchexec; print('OK from', benchexec.__file__)"

# 3) host z3
echo -n "host z3… "; which z3
echo -n "  version: "; z3 --version

# 4) container z3
echo -n "in SIF z3… "  
singularity exec ~/containers/z3-benchexec.sif z3 --version

# 5) SLURM‐mode dry-run in an interactive allocation:
echo
echo "Testing SLURM wrapper on a compute node via salloc…"
salloc -p compute-p1 -t 00:01:00 -c 1 --threads-per-core=1 --mem-per-cpu=2G bash -l -c '
  export PYTHONPATH=~/benchExec-dev:$PYTHONPATH
  python3 ~/benchExec-dev/contrib/slurm-benchmark.py \
    --slurm \
    --no-container \
    --no-hyperthreading \
    -d \
    -N 1 \
    -c 1 \
    vlsat3-benchmark.xml
'

echo
echo "If that completed (you saw a JobID and a “completed” message), your real SBATCH script will work."


