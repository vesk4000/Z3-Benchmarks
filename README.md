# Z3 Benchmarking with BenchExec

This repository contains a Docker-based setup for benchmarking multiple versions of Z3 using BenchExec.

## Features

- **Multiple Z3 versions**: Latest, 4.12.0, and 4.11.2
- **Proper cgroup setup**: Follows BenchExec container requirements
- **VS Code dev container**: Easy development environment
- **Security**: Runs as non-root user inside container

## Quick Start

1. **Open in VS Code Dev Container**:
   - Open this folder in VS Code
   - When prompted, click "Reopen in Container" or use Command Palette: "Dev Containers: Reopen in Container"
   - Wait for the container to build and start

2. **Verify the setup**:
   ```bash
   # Check Z3 versions are available
   z3-latest --version
   z3-4.12.0 --version
   z3-4.11.2 --version
   
   # Check BenchExec is working
   benchexec --help
   ```

## Container Structure

- **Z3 Versions**: Multiple Z3 versions installed in `/opt/z3-versions/`
  - `z3-latest`: Latest development version
  - `z3-4.12.0`: Stable release 4.12.0
  - `z3-4.11.2`: Older stable release 4.11.2

- **Symlinks**: Easy access via `/usr/local/bin/z3-*`

## Directory Structure

```
/workspace/
├── benchmarks/     # Your benchmark configurations
├── datasets/       # SMT problem files
├── results/        # Benchmark results
├── z3-*.py        # Tool definitions for different Z3 versions
└── sample-benchmark.xml  # Example benchmark configuration
```

## Adding SMT Datasets

1. Create directories under `datasets/` for different SMT-LIB logics:
   ```bash
   mkdir -p datasets/QF_LIA datasets/QF_LRA datasets/QF_BV
   ```

2. Add your `.smt2` files to the appropriate directories

3. The benchmark configuration will automatically pick them up

## Running Benchmarks

### Single Tool Execution with runexec

For simple measurements, use `runexec`:

```bash
# Test Z3 latest on a single file
runexec --walltimelimit 60s --memlimit 1GB --cores 0 \
  --output results/test.log -- z3-latest -smt2 datasets/QF_LIA/example.smt2

# Compare different versions
runexec --walltimelimit 60s --memlimit 1GB --cores 0 \
  -- z3-4.12.0 -smt2 datasets/QF_LIA/example.smt2
```

### Full Benchmark Suite with benchexec

For comprehensive benchmarking:

```bash
# Run the multi-version comparison
benchexec sample-benchmark.xml

# Run with specific number of CPU cores
benchexec sample-benchmark.xml --numOfThreads 4

# Generate HTML results
benchexec sample-benchmark.xml --outputpath results/
table-generator results/sample-benchmark.*.xml.bz2
```

## Customizing Benchmarks

Edit `sample-benchmark.xml` to:
- Add more Z3 configurations/options
- Include different SMT logic categories
- Adjust time/memory limits
- Add custom result columns

## Tool Definitions

Custom BenchExec tool definitions are provided:
- `z3-latest.py`: For the latest Z3 version
- `z3-4.12.0.py`: For Z3 4.12.0
- `z3-4.11.2.py`: For Z3 4.11.2

These handle result parsing and tool location for BenchExec.

## Container Requirements

This setup requires:
- Docker with `--privileged` access (for cgroup management)
- Sufficient memory (4GB+ recommended)
- Multiple CPU cores for parallel execution

## Troubleshooting

### Container Build Issues
- If the build fails, try rebuilding with: "Dev Containers: Rebuild Container"
- Check Docker has enough memory allocated (8GB+ recommended)

### BenchExec Issues
```bash
# Check cgroup setup
python3 -m benchexec.check_cgroups

# Check if Z3 versions are accessible
which z3-latest z3-4.12.0 z3-4.11.2
```

### Permission Issues
If you encounter permission issues:
```bash
sudo chown -R benchuser:benchuser /workspace
```

If BenchExec reports cgroup issues, verify the init script ran:
```bash
ls -la /sys/fs/cgroup/benchexec/
```

## Advanced Usage

### Adding More Z3 Versions
1. Modify the Dockerfile to add more versions
2. Create corresponding tool definition files
3. Update benchmark configurations

### Custom Z3 Configurations
Create tool definitions with different command-line options:
```python
def cmdline(self, executable, options, task, rlimits):
    return [executable, "-smt2", "-T:60", "model_validate=true"] + options + [task.single_input_file]
```

### Large-Scale Benchmarking
- Use BenchExec's parallel execution capabilities
- Consider resource limits carefully
- Monitor system load during execution

## Resources

- [BenchExec Documentation](https://github.com/sosy-lab/benchexec/blob/main/doc/INDEX.md)
- [SMT-LIB Benchmarks](https://www.smt-lib.org/)
- [Z3 Documentation](https://github.com/Z3Prover/z3)