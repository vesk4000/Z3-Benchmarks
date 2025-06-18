#!/usr/bin/env python3
# Generate SVG cactus, scatter, and critical-difference plots from a BenchExec CSV table
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from pathlib import Path
import csv
from Orange.evaluation import compute_CD, graph_ranks

RESULTS_FOLDER = Path("results/")
INPUT = "20250612_144947_vlsat3_a"
#INPUT = "20250612_144947_vlsat3_g"
#INPUT = "20250612_140513_smart_contracts"
#INPUT = "20250612_142520_smt-comp_2024"
#INPUT = "20250615_170652_parallel-hyperparameter-search"
OUTPUT = Path("../Writing/visualizations/")

def parse_results_csv(filepath: Path):
    """
    Parse BenchExec CSV table: three-row multi-index header.
    Returns dict of configs -> {filename: {'status', 'wall_time'}}
    """
    if filepath.suffix.lower() != '.csv':
        raise ValueError(f"Unsupported input format: {filepath}")
    # read table: auto-detect delimiter
    with open(filepath, newline='') as f:
        sample = f.read(2048)
        delim = csv.Sniffer().sniff(sample).delimiter
    df = pd.read_csv(filepath, sep=delim, header=[0,1,2], index_col=0)
    configs = df.columns.get_level_values(1).unique()
    data = {}
    for config in df.columns.get_level_values(1).unique():
        sub = df.xs(config, axis=1, level=1)
        # drop tool-name level
        sub.columns = sub.columns.droplevel(0)
        data[config] = {}
        for fname, row in sub.iterrows():
            status = row.get('status')
            wall = row.get('walltime (s)', np.nan)
            try:
                wall = float(wall)
            except:
                wall = np.nan
            data[config][fname] = {'status': status, 'wall_time': wall}
    return data

def parse_results_benchy(input_directory: Path):
    """
    Parse results from folder structure with .err files containing timing info.
    Returns dict of configs -> {filename: {'status', 'wall_time'}}
    """
    import re
    import glob
    
    data = {}
    
    # Find all .err files recursively in the input directory
    err_files = list(input_directory.rglob("*.err"))
    
    for err_file in err_files:
        # Parse filename to extract test name and solver config
        # Format: {test_file}_{solver_config}.err
        filename = err_file.stem  # Remove .err extension
        
        # Find the last underscore to split test file from solver config
        # This handles cases where test files might have underscores in their names
        parts = filename.split('_')
        if len(parts) < 2:
            continue  # Skip malformed filenames
        
        # Find where the solver config starts (look for z3- pattern)
        solver_start_idx = None
        for i, part in enumerate(parts):
            if part.endswith('.smt2'):
                solver_start_idx = i + 1
                break
        
        if solver_start_idx is None:
            continue  # Skip if no z3 config found
        
        test_file = '_'.join(parts[:solver_start_idx])
        solver_config = '_'.join(parts[solver_start_idx:])
        
        # Initialize solver config in data if not present
        if solver_config not in data:
            data[solver_config] = {}
        
        # Read and parse the .err file content
        try:
            with open(err_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for timeout patterns
            if 'CANCELLED AT' in content and 'DUE TO TIME LIMIT' in content:
                status = 'TIMEOUT'
                wall_time = np.nan
            elif 'srun: error:' in content and 'Terminated' in content:
                status = 'TIMEOUT'  
                wall_time = np.nan
            elif 'Job step aborted' in content:
                status = 'TIMEOUT'
                wall_time = np.nan
            elif 'srun: Force Terminated' in content:
                status = 'TIMEOUT'
                wall_time = np.nan
            else:
                # Look for timing information from the time command
                # Pattern: "Elapsed (wall clock) time (h:mm:ss or m:ss): X:XX.XX"
                time_match = re.search(r'Elapsed.*?(?:(\d+):)?(\d+):(\d+\.\d+)', content)
                if time_match:
                    # Parse the time components
                    hours = time_match.group(1)
                    minutes = int(time_match.group(2))
                    seconds = float(time_match.group(3))
                    
                    if hours:
                        hours = int(hours)
                        wall_time = hours * 3600 + minutes * 60 + seconds
                    else:
                        wall_time = minutes * 60 + seconds
                    
                    # Check exit status
                    exit_match = re.search(r'Exit status:\s*(\d+)', content)
                    if exit_match and exit_match.group(1) == '0':
                        status = 'SUCCESS'
                    else:
                        status = 'ERROR'
                else:
                    # No timing info found, might be an error
                    status = 'ERROR'
                    wall_time = np.nan
        
        except Exception as e:
            # Handle file reading errors
            status = 'ERROR'
            wall_time = np.nan
        
        # Store the result
        data[solver_config][test_file] = {
            'status': status,
            'wall_time': wall_time
        }
    
    return data

def plot_quantile(data, outdir):
    plt.figure(figsize=(10, 6))
    for cfg, results in data.items():
        # Only include successful runs, exclude timeouts and errors
        times = [v['wall_time'] for v in results.values() if v['status'] == 'SUCCESS' and not np.isnan(v['wall_time'])]
        if times:
            times_sorted = sorted(times)
            xs = np.arange(1, len(times_sorted)+1)
            plt.step(xs, times_sorted, where='post', label=cfg, linewidth=2)
            
            # Debug: check for any issues around position 35
            if len(times_sorted) > 35:
                print(f"DEBUG {cfg}: Position 30-40 times: {times_sorted[29:40]}")
    
    plt.xlabel('Number of instances solved')
    plt.ylabel('Wall time (s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y-axis scale options:
    # plt.yscale('log')  # Full logarithmic
    plt.yscale('symlog', linthresh=1)  # Symmetric log scale with linear threshold at 1
    # plt.yscale('linear')  # Linear scale (default)
    
    # X-axis scale options (uncomment if desired):
    # plt.xscale('log')  # Logarithmic x-axis (focuses on early solved problems)
    # plt.xscale('symlog', linthresh=10)  # Symmetric log x-axis
    
    plt.title('Quantile Plot (Cactus Plot)')
    
    # Ensure absolute path and overwrite existing file
    p = os.path.abspath(os.path.join(outdir, 'quantile.svg'))
    print(f"Saving quantile plot to: {p}")
    plt.savefig(p, format='svg', bbox_inches='tight')
    plt.close()

def plot_scatter(data, outdir):
    all_times = [v['wall_time'] for results in data.values() for v in results.values() if not np.isnan(v['wall_time'])]
    max_time = max(all_times) if all_times else 0
    for a, b in itertools.combinations(data.keys(), 2):
        x, y = [], []
        for fn in set(data[a]) & set(data[b]):
            ta = data[a][fn]['wall_time'] if data[a][fn]['status'] != 'TIMEOUT' else max_time
            tb = data[b][fn]['wall_time'] if data[b][fn]['status'] != 'TIMEOUT' else max_time
            x.append(ta)
            y.append(tb)
        plt.figure()
        plt.scatter(x, y, alpha=0.5)
        lim = max(max(x, default=0), max(y, default=0))
        plt.plot([0, lim], [0, lim], 'k--')
        plt.xlabel(f'Wall time ({a})')
        plt.ylabel(f'Wall time ({b})')
        plt.title(f'Scatter: {a} vs {b}')
        plt.grid(True)
        fname = f'scatter_{a.replace(" ","_")}_vs_{b.replace(" ","_")}.svg'
        plt.savefig(os.path.join(outdir, fname), format='svg')
        plt.close()

def plot_critical_difference(data, outdir):
    """Generate critical difference plot using Orange's graph_ranks function."""
    from scipy.stats import friedmanchisquare
    
    cfgs = list(data.keys())
    fns = sorted(set().union(*[set(d.keys()) for d in data.values()]))
    
    # Filter out test cases where we have no data at all for any solver
    valid_fns = []
    for fn in fns:
        has_any_data = False
        for cfg in cfgs:
            if fn in data[cfg]:
                has_any_data = True
                break
        if has_any_data:
            valid_fns.append(fn)
    
    if len(valid_fns) < 2:
        print("Not enough valid test cases for critical difference plot")
        return
    
    # Estimate timeout limit from the data (use max successful time + buffer)
    all_successful_times = []
    for cfg_data in data.values():
        for result in cfg_data.values():
            if result['status'] == 'SUCCESS' and not np.isnan(result['wall_time']):
                all_successful_times.append(result['wall_time'])
    
    if all_successful_times:
        timeout_penalty = max(all_successful_times) * 1.2  # 20% penalty over max successful time
    else:
        timeout_penalty = 3600  # Default 1 hour if no successful runs
    
    n = len(valid_fns)
    mat = np.zeros((len(cfgs), n))
    
    for i, cfg in enumerate(cfgs):
        for j, fn in enumerate(valid_fns):
            if fn in data[cfg]:
                result = data[cfg][fn]
                if result['status'] == 'SUCCESS' and not np.isnan(result['wall_time']):
                    mat[i, j] = result['wall_time']
                elif result['status'] == 'TIMEOUT':
                    mat[i, j] = timeout_penalty
                else:
                    # ERROR or invalid data - treat as worse than timeout
                    mat[i, j] = timeout_penalty * 1.1
            else:
                # No data for this test case - treat as worse than timeout
                mat[i, j] = timeout_penalty * 1.1
    
    # Compute ranks for each test case (lower time = better rank)
    ranks = np.array([pd.Series(mat[:, j]).rank(method='average').values for j in range(n)]).T
    mean_ranks = np.mean(ranks, axis=1)
    
    # Check if we have valid ranks
    if np.any(np.isnan(mean_ranks)):
        print("Invalid ranking data for critical difference plot")
        return
    
    print("=== FRIEDMAN TEST ===")
    print(f"Testing {len(cfgs)} solvers on {n} test cases")
    
    # Perform Friedman test
    # Each row in ranks.T corresponds to ranks for one test case across all solvers
    try:
        # Prepare data for Friedman test (each argument is the ranks for one solver across all test cases)
        solver_ranks = [ranks[i, :] for i in range(len(cfgs))]
        
        friedman_stat, friedman_p = friedmanchisquare(*solver_ranks)
        
        print(f"Friedman chi-square statistic: {friedman_stat:.4f}")
        print(f"p-value: {friedman_p:.6f}")
        print(f"Degrees of freedom: {len(cfgs) - 1}")
        
        alpha = 0.05
        if friedman_p < alpha:
            print(f"✓ SIGNIFICANT: p < {alpha}, rejecting null hypothesis")
            print("  There are significant differences between solvers")
            print("  → Proceeding with post-hoc critical difference analysis")
            proceed_with_cd = True
        else:
            print(f"✗ NOT SIGNIFICANT: p ≥ {alpha}, failing to reject null hypothesis")
            print("  No significant differences detected between solvers")
            print("  → Skipping critical difference plot (not statistically justified)")
            proceed_with_cd = False
            
        # Print mean ranks for interpretation
        print(f"\nMean ranks (lower = better):")
        for i, cfg in enumerate(cfgs):
            print(f"  {cfg}: {mean_ranks[i]:.2f}")
            
    except Exception as e:
        print(f"Error performing Friedman test: {str(e)}")
        print("Proceeding with critical difference plot anyway...")
        proceed_with_cd = True
    
    if not proceed_with_cd:
        return
    
    print(f"\n=== CRITICAL DIFFERENCE ANALYSIS ===")
    
    # Shorten config names for better display
    shortened_cfgs = []
    for cfg in cfgs:
        # Remove common prefixes and use abbreviations
        short_name = cfg.replace('z3-', '').replace('-and-', '&').replace('-sequential', '-seq')
        # Further abbreviations
        short_name = short_name.replace('bit-blasting', 'bit-blast')
        short_name = short_name.replace('lazy-bit-blast', 'lazy-bb')
        short_name = short_name.replace('sls', 'SLS')
        shortened_cfgs.append(short_name)
    
    try:
        cd = compute_CD(mean_ranks, n, alpha='0.05', test='nemenyi')
        output_path = os.path.join(outdir, 'critical_difference.svg')
        
        print(f"Critical difference threshold: {cd:.4f}")
        print("Pairs with |mean_rank_diff| > CD are significantly different")
        
        # Use Orange's graph_ranks function with adjusted parameters for better layout
        graph_ranks(mean_ranks, shortened_cfgs, cd=cd, 
                   width=8,        # Increase width to give more space
                   textspace=2.5,  # Increase text space for longer names
                   filename=output_path)
        print(f"Critical difference plot saved to: {output_path}")
        print(f"Config name mapping:")
        for orig, short in zip(cfgs, shortened_cfgs):
            print(f"  {orig} -> {short}")
        
        # Print some statistics about timeout handling
        timeout_counts = {}
        error_counts = {}
        for cfg in cfgs:
            timeouts = sum(1 for result in data[cfg].values() if result['status'] == 'TIMEOUT')
            errors = sum(1 for result in data[cfg].values() if result['status'] not in ['SUCCESS', 'TIMEOUT'])
            timeout_counts[cfg] = timeouts
            error_counts[cfg] = errors
        
        print(f"\nTimeout penalty time used: {timeout_penalty:.2f}s")
        print("Timeout counts per configuration:")
        for cfg in cfgs:
            print(f"  {cfg}: {timeout_counts[cfg]} timeouts, {error_counts[cfg]} errors")
        
    except Exception as e:
        print(f"Error generating critical difference plot: {str(e)}")
        # If the textspace parameter doesn't work, try without it
        try:
            graph_ranks(mean_ranks, shortened_cfgs, cd=cd, width=10, filename=output_path)
            print(f"Critical difference plot saved (fallback): {output_path}")
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")

def draw_quantile(ax, data):
    """Draw quantile plot on given axes without showing or saving."""
    for cfg, results in data.items():
        times = sorted([v['wall_time'] for v in results.values() if v['status'] != 'TIMEOUT'])
        if times:
            xs = np.arange(1, len(times)+1)
            ax.step(xs, times, where='post', label=cfg)
    ax.set_xlabel('Number of instances solved')
    ax.set_ylabel('Wall time (s)')
    ax.legend()
    ax.grid(True)

def draw_scatter(ax, data, a, b):
    """Draw scatter plot for two configs on given axes."""
    all_times = [v['wall_time'] for results in data.values() for v in results.values() if not np.isnan(v['wall_time'])]
    max_time = max(all_times) if all_times else 0
    x, y = [], []
    for fn in set(data[a]) & set(data[b]):
        ta = data[a][fn]['wall_time'] if data[a][fn]['status'] != 'TIMEOUT' else max_time
        tb = data[b][fn]['wall_time'] if data[b][fn]['status'] != 'TIMEOUT' else max_time
        x.append(ta)
        y.append(tb)
    ax.scatter(x, y, alpha=0.5)
    lim = max(max(x, default=0), max(y, default=0))
    ax.plot([0, lim], [0, lim], 'k--')
    ax.set_xlabel(f'Wall time ({a})')
    ax.set_ylabel(f'Wall time ({b})')
    ax.grid(True)

def draw_critical_difference(ax, data):
    """Simplified placeholder for critical difference in interactive plots."""
    # Since Orange's graph_ranks saves directly to file, we can't easily render into axes
    # For interactive plots, show a message directing to the saved file
    ax.text(0.5, 0.5, 'Critical Difference plot\nsaved separately as\ncritical_difference.svg', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def interactive_plots(data):
    """Show overview grid of all plots, then allow navigating to full-size zoomed views."""
    import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, numpy as np
    # prepare plot definitions
    plot_defs = []
    plot_defs.append(('Quantile Plot', lambda ax: draw_quantile(ax, data)))
    cfgs = list(data.keys())
    for a, b in itertools.combinations(cfgs, 2):
        plot_defs.append((f'Scatter: {a} vs {b}', lambda ax, a=a, b=b: draw_scatter(ax, data, a, b)))
    #plot_defs.append(('Critical Difference', lambda ax: draw_critical_difference(ax, data)))
    n = len(plot_defs)
    # determine grid size for overview
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(12, 8))
    # create grid axes for overview
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    overview_axes = []
    for i, (title, func) in enumerate(plot_defs):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        func(ax)
        ax.set_title(title, fontsize=9)
        overview_axes.append(ax)
    # create full-size axes (initially hidden)
    full_axes = []
    for title, func in plot_defs:
        # create a full-figure axes for zoomed view
        axf = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        func(axf)
        axf.set_title(title)
        axf.set_visible(False)
        full_axes.append(axf)
    # navigation state: 0 = overview, 1..n = full views
    state = {'i': 0}
    def show_state():
        # overview
        if state['i'] == 0:
            for ax in overview_axes: ax.set_visible(True)
            for ax in full_axes: ax.set_visible(False)
        else:
            for ax in overview_axes: ax.set_visible(False)
            for j, axf in enumerate(full_axes, start=1):
                axf.set_visible(j == state['i'])
        fig.canvas.draw_idle()
    def on_click(event):
        # click on overview thumbnail to zoom in, click anywhere in zoom to return overview
        if state['i'] == 0 and event.inaxes in overview_axes:
            idx = overview_axes.index(event.inaxes) + 1
            state['i'] = idx
        elif state['i'] != 0:
            state['i'] = 0
        show_state()
    fig.canvas.mpl_connect('button_press_event', on_click)
    show_state()
    plt.tight_layout()
    plt.show()

def main():
    input_directory = Path(RESULTS_FOLDER) / INPUT
    output_directory = Path(OUTPUT) / INPUT
    os.makedirs(output_directory, exist_ok=True)

    data = parse_results_benchy(input_directory)
    plot_quantile(data, output_directory)
    plot_scatter(data, output_directory)
    plot_critical_difference(data, output_directory)
    #interactive_plots(data)

if __name__ == '__main__':
    main()