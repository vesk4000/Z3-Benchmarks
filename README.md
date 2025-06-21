# Z3 Benchmarking on SLURM

A comprehensive benchmarking framework for evaluating Z3 solver configurations on SMT-LIB benchmarks using SLURM clusters.

## Quick Setup

1. **Clone and setup environment**:
   ```bash
   git clone <your-repo>
   cd Benchmarking
   chmod +x init.sh *.sh
   ./init.sh  # Sets up conda environment and dependencies
   ```

2. **Download benchmark datasets**:
   ```bash
   mkdir -p datasets
   
   # QF_BV dataset from Zenodo
   cd datasets
   wget -O QF_BV.tar.zst "https://zenodo.org/records/11061097/files/QF_BV.tar.zst?download=1"
   tar -I zstd -xf QF_BV.tar.zst
   rm QF_BV.tar.zst
   
   # VLSAT3 dataset from CADP
   mkdir -p VLSAT3
   wget -r --no-parent -A "vlsat3*.smt2.bz2" https://cadp.inria.fr/ftp/benchmarks/vlsat/ -P VLSAT3
   find VLSAT3 -name "*.smt2.bz2" -exec bunzip2 {} +
   
   # SMT-COMP 2024 dataset (adjust URL/path as needed)
   # Download from SMT-COMP official sources or your institution's mirror
   ```

3. **Update SLURM configuration** in scripts (partition, account, paths):
   ```bash
   # Edit slurm-*.sh files to match your cluster settings:
   # - #SBATCH --partition=your-partition
   # - #SBATCH --account=your-account  
   # - cd /your/scratch/path/Benchmarking
   ```

4. **Update dataset paths** in `plot_results.py`:
   ```python
   # Edit the DATASETS_FOLDER variable to match your setup:
   DATASETS_FOLDER = Path("/path/to/your/datasets/")
   ```

## Dataset Requirements

The framework expects datasets in these locations:
- `datasets/QF_BV/` - SMT-LIB QF_BV benchmarks
- `datasets/VLSAT3/` - VLSAT3 benchmarks  
- `datasets/SMT-COMP_2024/` - SMT-COMP 2024 benchmarks
- `datasets/Smart_Contract_Verification/` - Smart contract benchmarks

**Required tools for dataset extraction:**
- `zstd` - for QF_BV.tar.zst extraction
- `bzip2` - for VLSAT3 .bz2 files
- `wget` - for downloading datasets

Install on Ubuntu/Debian:
```bash
sudo apt-get install zstd bzip2 wget
```

## Available Benchmarks

The framework includes pre-configured benchmark suites:

| Name | Dataset | Configs | Time Limit | Description |
|------|---------|---------|------------|-------------|
| `smt-comp_2024` | SMT-COMP 2024 | 4 solver configs | 20 min | Main competition benchmark |
| `vlsat3_a` | VLSAT3 | 4 solver configs | 40 min | VLSAT3 'a' instances |
| `vlsat3_g` | VLSAT3 | 4 solver configs | 40 min | VLSAT3 'g' instances |
| `smart_contracts` | Smart contracts | 4 solver configs | 60 min | Smart contract verification |
| `parallel-hyperparameter-search` | SMT-COMP 2024 | 500 random configs | 4 min | Hyperparameter optimization |
| `parallel-scaling-X` | SMT-COMP 2024 | 1 config | 20 min | Scaling analysis (X = 2,4,8,16,32,64 cores) |

## Running Benchmarks

### Standard Benchmarks
```bash
# Submit specific benchmark jobs
sbatch slurm-smt-comp_2024.sh
sbatch slurm-vlsat3_a.sh
sbatch slurm-vlsat3_g.sh
sbatch slurm-smart_contracts.sh
```

### Hyperparameter Search
```bash
sbatch slurm-parallel-hyperparameter-search.sh
```

### Scaling Analysis
```bash
# Run parallel scaling experiments
sbatch slurm-parallel-scaling-2.sh
sbatch slurm-parallel-scaling-4.sh
sbatch slurm-parallel-scaling-8.sh
sbatch slurm-parallel-scaling-16.sh
sbatch slurm-parallel-scaling-32.sh
sbatch slurm-parallel-scaling-64.sh
```

### Monitor Jobs
```bash
squeue -u $USER                    # Check job status
scancel <job-id>                   # Cancel specific job
tail -f logs/<job-name>-<id>.out   # Follow job output
```

## Solver Configurations

The framework tests these Z3 configurations:

- **z3-bit-blast**: Bit-blasting with SMT tactic
- **z3-int-blast**: Integer solver with BV solver=2  
- **z3-lazy-bit-blast**: Polynomial arithmetic with BV solver=1
- **z3-sls-and-bit-blasting-sequential**: Stochastic local search (sequential)

For parallel experiments, additional configurations test various thread combinations and parameter settings.

## Analysis & Visualization

1. **Process results** (automatically finds latest results):
   ```bash
   python plot_results.py  # Edit INPUT variable for specific dataset
   ```

2. **Sort hyperparameter results**:
   ```bash
   python sort_hyperparameter_results.py results/TIMESTAMP_*/parallel_hyperparameter_search_ranks.txt
   ```

3. **Generated plots**:
   - `quantile.svg`: Cactus plot showing solver performance
   - `scatter_*.svg`: Pairwise solver comparisons  
   - `critical_difference.svg`: Statistical significance analysis
   - `histogram_family_binned_performance.svg`: Performance by benchmark family

4. **Critical difference analysis**:
   ```bash
   python sort_hyperparameter_results.py ranks.txt --cd-only --instances 30
   ```

## Key Files

| File | Purpose |
|------|---------|
| `benchy.py` | Main benchmarking script with solver configurations |
| `slurm-*.sh` | SLURM job scripts for different experiments |
| `plot_results.py` | Visualization and statistical analysis |
| `sort_hyperparameter_results.py` | Results processing and ranking |
| `init.sh` | Environment setup script |
| `*_tasks*.txt` | Task lists for different benchmark subsets |

## Configuration

### Solver Configs (in benchy.py)
```python
four_horsemen = [
    {
        "name": "z3-bit-blast", 
        "command": ["z3", "-smt2", "tactic.default_tactic='...'"]
    },
    # Add custom configurations here
]
```

### SLURM Resources (in slurm-*.sh)
```bash
#SBATCH --time=02:00:00      # Adjust time limit
#SBATCH --mem-per-cpu=3200MB # Adjust memory
#SBATCH --ntasks=64          # Adjust parallelism
```

### Dataset Paths (in plot_results.py)
```python
DATASETS_FOLDER = Path("D:/datasets/")  # Update to your path
INPUT = "20250618_214918_parallel-hyperparameter-search"  # Select dataset
```

## Output Structure

```
results/
  TIMESTAMP_experiment-name/
    dataset/family/instance.smt2_config.{out,err}
    logs/
cache/
  cleaned_data_*.pkl        # Processed results cache
  smt2_stats_cache.json     # SMT2 file analysis cache
logs/
  job-name-id.{out,err}     # SLURM job logs
```

## Advanced Usage

### Custom Task Lists
Create text files with instance paths:
```bash
echo "SMT-COMP_2024/QF_BV/family/instance.smt2" > custom_tasks.txt
python benchy.py --name custom --task-list custom_tasks.txt
```

### Merging Multiple Runs
The framework automatically merges results from multiple runs of the same experiment, taking the best result for each instance.

### Memory and Time Limits
Each benchmark configuration specifies appropriate limits. Smart contracts use higher memory (12GB) and longer timeouts (60min).

## Troubleshooting

- **Permission denied**: Run `chmod +x *.sh` to make scripts executable
- **Module not found**: Check conda environment activation in SLURM scripts
- **Out of memory**: Reduce `--mem-per-cpu` or increase memory limits in job scripts
- **Jobs pending**: Check cluster queue status and resource availability
- **No results**: Verify dataset paths and task file contents


