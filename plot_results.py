#!/usr/bin/env python3
# Generate SVG cactus, scatter, and critical-difference plots from a BenchExec CSV table
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import itertools
import pandas as pd
from pathlib import Path
import csv
import re
import json
import hashlib
from Orange.evaluation import compute_CD, graph_ranks
import copy
import pickle

RESULTS_FOLDER = Path("results/")
DATASETS_FOLDER = Path("D:/datasets/")
INPUT = "20250618_214918_parallel-hyperparameter-search"
#INPUT = "20250612_144947_vlsat3_g"
#INPUT = "20250612_140513_smart_contracts"
#INPUT = "20250612_142520_smt-comp_2024"
#INPUT = "2077_smt-comp_2024"
#INPUT = "20250618_151400_smt-comp_2024"
#INPUT = "20250615_170652_parallel-hyperparameter-search"
OUTPUT = Path("../Writing/visualizations/")
INCLUDE_TIMEOUT = True
PENALTY_TIME_SECONDS = 20 * 60

# Legend position configuration
# Options: 'best', 'upper right', 'upper left', 'lower left', 'lower right', 
#          'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
# Or use bbox_to_anchor with tuple like (1.05, 1) for outside positioning
LEGEND_CONFIG = {
    'quantile': 'best',                    # Quantile/cactus plot
    'scatter': 'upper left',               # Scatter plots
    'binned_performance': (1.05, 1),      # Binned performance histograms (outside plot)
    'family_performance': (1.05, 1),      # Family performance histogram (outside plot)
}

# Cache configuration
CACHE_DIR = Path("cache")
SMT2_STATS_CACHE_FILE = CACHE_DIR / "smt2_stats_cache.json"

def ensure_cache_dir():
    """Ensure cache directory exists."""
    CACHE_DIR.mkdir(exist_ok=True)

def load_smt2_cache():
    """Load SMT2 statistics cache from disk."""
    if not SMT2_STATS_CACHE_FILE.exists():
        return {}
    try:
        with open(SMT2_STATS_CACHE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load SMT2 cache: {e}")
        return {}

def save_smt2_cache(cache):
    """Save SMT2 statistics cache to disk."""
    ensure_cache_dir()
    try:
        with open(SMT2_STATS_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save SMT2 cache: {e}")

def save_cleaned_data_cache(data, cache_file_path):
    """Save cleaned data to cache file using pickle."""
    try:
        ensure_cache_dir()
        with open(cache_file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved cleaned data cache to: {cache_file_path}")
    except Exception as e:
        print(f"Warning: Could not save cleaned data cache: {e}")

def load_cleaned_data_cache(cache_file_path):
    """Load cleaned data from cache file using pickle."""
    if not cache_file_path.exists():
        return None
    
    try:
        with open(cache_file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded cleaned data cache from: {cache_file_path}")
        return data
    except Exception as e:
        print(f"Warning: Could not load cleaned data cache: {e}")
        return None

def get_cleaned_data_cache_path(input_names):
    """Generate cache file path based on input dataset names."""
    cache_name = "cleaned_data_" + "_".join(input_names) + ".pkl"
    return CACHE_DIR / cache_name

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

def parse_smt2_file(smt2_path: Path):
    """
    Parse SMT2 file to extract various statistics.
    Returns dict with bit-width stats, operation ratios, category, and expected status.
    """
    if not smt2_path.exists():
        return None
    
    try:
        with open(smt2_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading SMT2 file {smt2_path}: {e}")
        return None
    
    stats = {
        'bit_widths': [],
        'variables': 0,
        'category': None,
        'expected_status': None
    }
    
    # Extract category from set-info
    category_match = re.search(r'\(set-info\s+:category\s+"([^"]+)"\)', content)
    if category_match:
        stats['category'] = category_match.group(1)
    
    # Extract expected status
    status_match = re.search(r'\(set-info\s+:status\s+(\w+)\)', content)
    if status_match:
        stats['expected_status'] = status_match.group(1)
    
    # Look for all (_ BitVec x) patterns to collect bit-widths
    bitvec_type_pattern = r'\(\s*_\s+BitVec\s+(\d+)\s*\)'
    bitvec_type_matches = re.findall(bitvec_type_pattern, content, re.IGNORECASE)
    for width_str in bitvec_type_matches:
        stats['bit_widths'].append(int(width_str))
    
    # Count variables and functions (both declare-const and declare-fun)
    declare_const_pattern = r'\(declare-const\s+\w+'
    declare_fun_pattern = r'\(declare-fun\s+\w+'
    const_matches = re.findall(declare_const_pattern, content)
    fun_matches = re.findall(declare_fun_pattern, content)
    stats['variables'] = len(const_matches) + len(fun_matches)
    
    # Look for bit-vector literals to get more bit-widths
    bitvec_literal_pattern = r'#b([01]+)'
    bitvec_matches = re.findall(bitvec_literal_pattern, content)
    for bits in bitvec_matches:
        stats['bit_widths'].append(len(bits))
    
    # Look for hex bit-vector literals #x...
    hex_bitvec_pattern = r'#x([0-9a-fA-F]+)'
    hex_matches = re.findall(hex_bitvec_pattern, content)
    for hex_str in hex_matches:
        # Each hex digit represents 4 bits
        stats['bit_widths'].append(len(hex_str) * 4)
    
    # Count boolean operations
    boolean_ops = ['and', 'or', 'xor', 'not', 'distinct', '=', '=>', 'ite']
    for op in boolean_ops:
        # Use word boundaries to avoid matching parts of other operations
        pattern = r'\(\s*' + re.escape(op) + r'\s+'
        matches = re.findall(pattern, content)
        stats[op] = len(matches)
    
    # Count bit-vector operations
    bitvec_ops = ['bvand', 'bvor', 'bvxor', 'bvnot', 'bvadd', 'bvsub', 'bvmul', 
                  'bvudiv', 'bvurem', 'bvshl', 'bvlshr', 'bvashr', 'bvult', 
                  'bvule', 'bvugt', 'bvuge', 'bvslt', 'bvsle', 'bvsgt', 'bvsge',
                  'bvneg', 'bvsdiv', 'bvsrem', 'bvsmod', 'bvnand', 'bvnor',
                  'bvcomp', 'concat', 'extract']

    for op in bitvec_ops:
        pattern = r'\(\s*' + re.escape(op) + r'\s+'
        matches = re.findall(pattern, content)
        stats[op] = len(matches)
    
    # Count all assertions for a different perspective on operations
    assert_pattern = r'\(assert\s+'
    assert_matches = re.findall(assert_pattern, content)
    stats['assertions'] = len(assert_matches)
    
    return stats

def compute_smt2_statistics(stats: dict):
    """
    Compute derived statistics from parsed SMT2 data.
    Returns dict with min/max/mean/total bit-widths and operation ratios.
    """
    computed = copy.deepcopy(stats)
    bit_widths = computed.pop('bit_widths')
    
    computed['min_bit_width'] = min(bit_widths)
    computed['max_bit_width'] = max(bit_widths)
    computed['avg_bit_width'] = np.mean(bit_widths)
    computed['median_bit_width'] = np.median(bit_widths)
    computed['total_bit_width'] = sum(bit_widths)
    computed['category'] = stats['category']
    computed['expected_status'] = stats['expected_status']
    
    return computed

def get_actual_result_from_out_file(out_file_path: Path):
    """
    Parse the .out file to get the actual solver result (sat/unsat/unknown).
    """
    if not out_file_path.exists():
        return None
    
    try:
        with open(out_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip().lower()
        
        if 'sat' in content and 'unsat' not in content:
            return 'sat'
        elif 'unsat' in content:
            return 'unsat'
        elif 'unknown' in content:
            return 'unknown'
        else:
            return 'error'  # No clear result found
    except Exception as e:
        print(f"Error reading .out file {out_file_path}: {e}")
        return None

def find_smt2_file(result_file_path: Path, datasets_folder: Path = DATASETS_FOLDER):
    """
    Given a result file path, find the corresponding SMT2 file in the datasets folder.
    
    Example:
    Input: results/20250612_144947_vlsat3_g/VLSAT3/cadp.inria.fr/ftp/benchmarks/vlsat/vlsat3_g00.smt2_z3-bit-blast.err
    Output: datasets/VLSAT3/cadp.inria.fr/ftp/benchmarks/vlsat/vlsat3_g00.smt2
    """
    # Get the relative path from the results folder
    path_parts = result_file_path.parts
    datasets_folder = DATASETS_FOLDER
    
    # Find where the dataset name starts (after the timestamp folder)
    dataset_start_idx = None
    for i, part in enumerate(path_parts):
        if part.startswith('20'):  # timestamp folder
            dataset_start_idx = i + 1
            break
    
    if dataset_start_idx is None:
        return None
    
    # Reconstruct the path within the dataset
    dataset_path_parts = path_parts[dataset_start_idx:]
    
    # The last part is the result filename, we need to extract the SMT2 filename
    result_filename = dataset_path_parts[-1]
    
    # Extract SMT2 filename (everything before the first underscore that's followed by solver config)
    # Pattern: vlsat3_g00.smt2_z3-bit-blast.err -> vlsat3_g00.smt2
    match = re.match(r'(.+\.smt2)_.*', result_filename)
    if match:
        smt2_filename = match.group(1)
    else:
        # Fallback: assume the SMT2 file has the same base name
        smt2_filename = result_filename.replace('.err', '').replace('.out', '')
        if not smt2_filename.endswith('.smt2'):
            smt2_filename += '.smt2'
    
    # Construct the full path to the SMT2 file
    smt2_path = datasets_folder / Path(*dataset_path_parts[:-1]) / smt2_filename
    
    return smt2_path

def parse_results_benchy(input_directory: Path):
    """
    Parse results from folder structure with .err files containing timing info.
    Returns dict of configs -> {filename: {'status', 'wall_time', 'smt2_stats', 'actual_result'}}
    """
    
    data = {}
    
    # Find all .err files recursively in the input directory
    err_files = list(input_directory.rglob("*.err"))
    
    print(f"Found {len(err_files)} .err files to process")
    
    # First pass: collect all unique SMT2 files that need to be parsed
    smt2_files_to_parse = set()
    err_to_smt2_mapping = {}
    
    for err_file in err_files:
        smt2_path = find_smt2_file(err_file)
        assert smt2_path
        smt2_files_to_parse.add(smt2_path)
        err_to_smt2_mapping[err_file] = smt2_path

    print(f"Found {len(smt2_files_to_parse)} unique SMT2 files to parse")
    
    cache = load_smt2_cache()
    
    for i, smt2_path in enumerate(smt2_files_to_parse):
        if i % 50 == 0:
            print(f"Parsing SMT2 files: {i}/{len(smt2_files_to_parse)}")
        
        path_str = str(smt2_path.resolve())

        alt_path_formats = [
            str(smt2_path),  # Original relative format
            str(smt2_path.absolute()),  # Absolute format
            str(smt2_path).replace('\\', '/'),  # Forward slash format
            str(smt2_path.resolve()).replace('\\', '/')  # Absolute forward slash
        ]
        
        stats = None
        for alt_path in alt_path_formats:
            if alt_path in cache:
                stats = cache[alt_path].get('stats')
                break
        else:
            stats = compute_smt2_statistics(parse_smt2_file(smt2_path))
            cache[path_str] = {
                'hash': "Cowabunga",
                'stats': stats,
                'timestamp': str(pd.Timestamp.now())
            }
    
    # Force save the cache after parsing all files
    save_smt2_cache(cache)
    
    # Third pass: process all .err files and assign pre-computed SMT2 stats
    for i, err_file in enumerate(err_files):
        if i % 100 == 0:
            print(f"Processing result files: {i}/{len(err_files)}")
        
        # Parse filename to extract test name and solver config
        filename = err_file.stem  # Remove .err extension
        
        # Find the last underscore to split test file from solver config
        parts = filename.split('_')
        assert len(parts) >= 2
            
        
        solver_start_idx = None
        for j, part in enumerate(parts):
            if part.endswith('.smt2'):
                solver_start_idx = j + 1
                break
        
        assert solver_start_idx is not None
        
        test_file = '_'.join(parts[:solver_start_idx])
        solver_config = '_'.join(parts[solver_start_idx:])
        
        # Extract the relative path from the .err file to preserve directory structure
        # Get the path relative to the input directory to preserve family structure
        relative_path = err_file.relative_to(input_directory)
        # Remove the filename and reconstruct the path with SMT2 file
        path_parts = list(relative_path.parts[:-1])  # Remove filename
        test_file_with_path = '/'.join(path_parts + [test_file])
        
        # Initialize solver config in data if not present
        if solver_config not in data:
            data[solver_config] = {}
        
        # Read and parse the .err file content
        with open(err_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if 'Out Of Memory' in content:
            status = 'MEMORY_LIMIT'
            wall_time = np.nan
        elif 'Command being timed' in content:
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
                assert False
        elif 'CANCELLED AT' in content and 'DUE TO TIME LIMIT' in content:
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
        elif 'expired' in content:
            status = "EXPIRED"
            wall_time = np.nan
        else:
            status = "BIG_ERROR"
            wall_time = np.nan
            
        
        # Get pre-computed SMT2 stats
        assert err_file in err_to_smt2_mapping
        smt2_path = err_to_smt2_mapping[err_file]
        smt2_stats = cache.get(str(smt2_path.resolve()).replace('/', '\\\\'))
        
        # Get actual result from .out file
        out_file = err_file.with_suffix('.out')
        actual_result = get_actual_result_from_out_file(out_file)
        
        assert smt2_stats['stats'] is not None

        # Store the result using the path-preserving test file name
        data[solver_config][test_file_with_path] = {
            'status': status,
            'wall_time': wall_time,
            'smt2_stats': smt2_stats['stats'],
            'actual_result': actual_result
        }
    
    # Print summary statistics
    print(f"\n=== SMT2 Statistics Summary ===")
    total_files = sum(len(results) for results in data.values())
    files_with_smt2_stats = sum(1 for results in data.values() 
                               for result in results.values() 
                               if result['smt2_stats'] is not None)
    
    print(f"Total result files processed: {total_files}")
    print(f"Files with SMT2 statistics: {files_with_smt2_stats}")
    
    return data

def clean_data(data, includeTimeout=True):
    """
    Clean the data by removing instances that:
    1. Don't exist for all configurations
    2. Have any configuration with invalid status (ERROR, EXPIRED, etc.) - only keep SUCCESS/TIMEOUT
    
    Args:
        data: Results data dict of configs -> {filename: {'status', 'wall_time', 'smt2_stats', 'actual_result'}}
    
    Returns:
        cleaned_data: Cleaned data with same structure
        removed_count: Number of instances removed
    """
    if not data:
        return data, 0
    
    configs = list(data.keys())
    
    # Collect all test files across all configurations
    all_test_files = set()
    for cfg_results in data.values():
        all_test_files.update(cfg_results.keys())
    
    print(f"Total unique test files found: {len(all_test_files)}")
    print(f"Configurations: {configs}")
    
    # Filter instances that exist for all configurations
    complete_instances = []
    for test_file in all_test_files:
        exists_in_all = True
        for cfg in configs:
            if test_file not in data[cfg]:
                exists_in_all = False
                break
        
        if exists_in_all:
            complete_instances.append(test_file)
    
    print(f"Instances that exist for all configurations: {len(complete_instances)}")
    missing_instances_removed = len(all_test_files) - len(complete_instances)
    
    # Filter instances with only valid status types (SUCCESS or TIMEOUT only)
    valid_instances = []
    invalid_count = 0
    
    for test_file in complete_instances:
        statuses = []
        for cfg in configs:
            status = data[cfg][test_file]['status']
            statuses.append(status)
        

        # Check if all statuses are either SUCCESS or TIMEOUT (no ERROR, EXPIRED, etc.)
        valid_statuses = all(status in ['SUCCESS', 'TIMEOUT' if includeTimeout else 'Cowabunga', 'MEMORY_LIMIT' if includeTimeout else 'Cowabunga'] for status in statuses)
        
        if valid_statuses:
            valid_instances.append(test_file)
        else:
            invalid_count += 1
            # Debug: show what invalid statuses look like
            if invalid_count <= 5:  # Only show first 5 examples
                print(f"  Invalid status example '{test_file}': {dict(zip(configs, statuses))}")
    
    print(f"Instances with only valid status types (SUCCESS/TIMEOUT): {len(valid_instances)}")
    print(f"Instances removed due to invalid status: {invalid_count}")
    
    # Create cleaned data
    cleaned_data = {}
    for cfg in configs:
        cleaned_data[cfg] = {}
        for test_file in valid_instances:
            cleaned_data[cfg][test_file] = data[cfg][test_file]
    
    total_removed = missing_instances_removed + invalid_count
    
    print(f"\n=== DATA CLEANING SUMMARY ===")
    print(f"Original instances: {len(all_test_files)}")
    print(f"Removed due to missing in some configs: {missing_instances_removed}")
    print(f"Removed due to invalid status (ERROR/EXPIRED/etc): {invalid_count}")
    print(f"Total removed: {total_removed}")
    print(f"Final clean instances: {len(valid_instances)}")
    print(f"Cleaning efficiency: {len(valid_instances)/len(all_test_files)*100:.1f}%")
    
    return cleaned_data, total_removed

def merge_and_clean_multiple_datasets(datasets, includeTimeout=True):
    """
    Merge multiple datasets intelligently, taking the best result for each test instance.
    
    For each test instance:
    - If it only appears in one dataset, include it
    - If it appears in multiple datasets:
      - Prioritize SUCCESS over TIMEOUT over ERROR/other statuses
      - For multiple SUCCESS results, take the one with shortest wall_time
      - For multiple TIMEOUT results, take any one (they're equivalent)
    
    Args:
        datasets: List of data dicts, each with structure: configs -> {filename: {'status', 'wall_time', 'smt2_stats', 'actual_result'}}
        includeTimeout: Whether to include TIMEOUT results in final dataset
    
    Returns:
        tuple: (merged_cleaned_data, merge_stats)
    """
    if not datasets:
        return {}, {}
    
    print(f"\n=== MERGING {len(datasets)} DATASETS ===")
    
    # First, collect all unique configurations and test files across all datasets
    all_configs = set()
    all_test_files = set()
    
    for i, dataset in enumerate(datasets):
        dataset_configs = set(dataset.keys())
        dataset_test_files = set()
        for config_results in dataset.values():
            dataset_test_files.update(config_results.keys())
        
        all_configs.update(dataset_configs)
        all_test_files.update(dataset_test_files)
        
        print(f"Dataset {i+1}: {len(dataset_configs)} configs, {len(dataset_test_files)} test files")
    
    print(f"Total unique configs: {len(all_configs)}")
    print(f"Total unique test files: {len(all_test_files)}")
    
    # Initialize merged dataset structure
    merged_data = {}
    for config in all_configs:
        merged_data[config] = {}
    
    # Track merge statistics
    merge_stats = {
        'test_files_processed': 0,
        'single_dataset_instances': 0,
        'multi_dataset_instances': 0,
        'success_over_timeout': 0,
        'success_over_error': 0,
        'faster_success_chosen': 0,
        'timeout_over_error': 0
    }
    
    # For each test file, find the best result across all datasets
    for test_file in all_test_files:
        merge_stats['test_files_processed'] += 1
        
        # Collect all results for this test file across datasets and configs
        test_file_results = {}  # config -> list of (dataset_idx, result)
        
        for config in all_configs:
            test_file_results[config] = []
            
            for dataset_idx, dataset in enumerate(datasets):
                if config in dataset and test_file in dataset[config]:
                    result = dataset[config][test_file]
                    test_file_results[config].append((dataset_idx, result))
        
        # Track how many datasets this test file appears in
        datasets_containing_file = set()
        for config_results in test_file_results.values():
            for dataset_idx, _ in config_results:
                datasets_containing_file.add(dataset_idx)
        
        if len(datasets_containing_file) == 1:
            merge_stats['single_dataset_instances'] += 1
        else:
            merge_stats['multi_dataset_instances'] += 1
        
        # For each config, choose the best result for this test file
        for config in all_configs:
            results = test_file_results[config]
            
            if not results:
                # This config doesn't have this test file in any dataset
                continue
            
# if len(results) == 1:
            #     # Only one result available, use it
            #     _, best_result = results[0]
            # else:
            # Multiple results available, choose the best one
            best_result = choose_best_result(results, merge_stats)
            if best_result == None:
                continue
            
            # Store the best result
            merged_data[config][test_file] = best_result
    
    print(f"\n=== MERGE STATISTICS ===")
    print(f"Test files processed: {merge_stats['test_files_processed']}")
    print(f"Single dataset instances: {merge_stats['single_dataset_instances']}")
    print(f"Multi-dataset instances: {merge_stats['multi_dataset_instances']}")
    print(f"SUCCESS chosen over TIMEOUT: {merge_stats['success_over_timeout']}")
    print(f"SUCCESS chosen over ERROR: {merge_stats['success_over_error']}")
    print(f"Faster SUCCESS chosen: {merge_stats['faster_success_chosen']}")
    print(f"TIMEOUT chosen over ERROR: {merge_stats['timeout_over_error']}")
    
    # Now apply the original cleaning logic to the merged dataset
    print(f"\n=== APPLYING STANDARD CLEANING TO MERGED DATA ===")
    cleaned_merged_data, removed_count = clean_data(merged_data, includeTimeout)
    
    return cleaned_merged_data, merge_stats

def choose_best_result(results, merge_stats):
    """
    Choose the best result from multiple results for the same test file and config.
    
    Priority order:
    1. SUCCESS with shortest wall_time
    2. TIMEOUT (any)
    3. ERROR/other statuses (any)
    
    Args:
        results: List of (dataset_idx, result) tuples
        merge_stats: Stats dict to update
    
    Returns:
        best_result: The chosen result dict
    """
    # Separate results by status type
    success_results = []
    timeout_results = []
    error_results = []
    
    for dataset_idx, result in results:
        status = result['status']
        if status == 'SUCCESS':
            success_results.append((dataset_idx, result))
        elif status == 'TIMEOUT': # TODO: Remove the MEMORY_LIMITs from the results
            timeout_results.append((dataset_idx, result))
        else:
            error_results.append((dataset_idx, result))
    
    # Choose best result based on priority
    if success_results:
        if len(success_results) == 1:
            _, best_result = success_results[0]
        else:
            # Multiple SUCCESS results, choose the one with shortest wall_time
            best_dataset_idx, best_result = min(success_results, 
                                               key=lambda x: x[1]['wall_time'] if not np.isnan(x[1]['wall_time']) else float('inf'))
            merge_stats['faster_success_chosen'] += 1
        
        # Track what we chose SUCCESS over
        if timeout_results:
            merge_stats['success_over_timeout'] += 1
        if error_results:
            merge_stats['success_over_error'] += 1
            
    elif timeout_results:
        # No SUCCESS, take any TIMEOUT
        _, best_result = timeout_results[0]
        
        if error_results:
            merge_stats['timeout_over_error'] += 1
            
    else:
        # Only ERROR results, take any one
        _, best_result = error_results[0]
        return None
    
    return best_result

def filter_task_file_remove_solved(task_file_path, cleaned_data, output_file_path, keep_solved=False):
    """
    Read a task file and create a new file.
    
    Args:
        task_file_path: Path to the original task file (e.g., SMT-COMP_2024_tasks_really_all.txt)
        cleaned_data: The cleaned data dict from clean_data() function
        output_file_path: Path for the new filtered task file
        keep_solved: If False, remove instances in cleaned data. If True, keep only instances in cleaned data.
    
    Returns:
        tuple: (original_count, kept_count, removed_count)
    """
    
    # Extract all test files that remain in cleaned data
    instances_in_cleaned_data = set()
    if cleaned_data:
        # Get all test files from the first configuration (they should be the same across all configs after cleaning)
        first_config = list(cleaned_data.keys())[0]
        instances_in_cleaned_data = set(cleaned_data[first_config].keys())
    
    print(f"Found {len(instances_in_cleaned_data)} instances in cleaned data")
    
    # Read the original task file
    try:
        with open(task_file_path, 'r', encoding='utf-8') as f:
            original_tasks = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading task file {task_file_path}: {e}")
        return 0, 0, 0
    
    print(f"Original task file contains {len(original_tasks)} instances")
    
    # Filter the instances based on the keep_solved flag
    filtered_tasks = []
    removed_count = 0
    
    for task in original_tasks:
        task = task.strip()
        if not task:
            continue
            
        # Check if this task exists in cleaned data
        is_in_cleaned_data = task in instances_in_cleaned_data
        
        if keep_solved:
            # Keep only instances that ARE in cleaned data
            if is_in_cleaned_data:
                filtered_tasks.append(task)
            else:
                removed_count += 1
        else:
            # Keep only instances that are NOT in cleaned data (original behavior)
            if not is_in_cleaned_data:
                filtered_tasks.append(task)
            else:
                removed_count += 1
    
    # Write the filtered task file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for task in filtered_tasks:
                f.write(task + '\n')
        print(f"Filtered task file saved to: {output_file_path}")
    except Exception as e:
        print(f"Error writing filtered task file {output_file_path}: {e}")
        return len(original_tasks), len(filtered_tasks), removed_count
    
    print(f"\n=== TASK FILE FILTERING SUMMARY ===")
    print(f"Original instances: {len(original_tasks)}")
    if keep_solved:
        print(f"Instances in cleaned data (kept): {len(filtered_tasks)}")
        print(f"Instances not in cleaned data (removed): {removed_count}")
    else:
        print(f"Instances in cleaned data (removed): {removed_count}")
        print(f"Challenging instances (kept): {len(filtered_tasks)}")
    print(f"Filtering efficiency: {len(filtered_tasks)/len(original_tasks)*100:.1f}% kept")
    
    return len(original_tasks), len(filtered_tasks), removed_count

def get_short_config_name(cfg):
    """
    Convert configuration names to short display names for consistency across all plots.
    """
    # Remove common prefixes and use abbreviations
    short_name = cfg.replace('z3-', '').replace('-and-', '&')
    # Further abbreviations
    short_name = short_name.replace('bit-blasting', 'bit-blast')
    short_name = short_name.replace('lazy-bit-blast', 'polysat')
    if 'sequential' in short_name:
        short_name = 'sls'
    return short_name

def get_config_style_mapping(configs):
    """
    Get consistent color and marker mappings for configurations.
    Returns dict with 'colors', 'markers', and 'short_names' keys.
    """
    # Define consistent colors and markers for the main configurations
    # Using distinct, colorblind-friendly colors
    style_map = {
        'bit-blast': {'color': '#1f77b4', 'marker': 'o'},      # Blue, circle
        'polysat': {'color': '#ff7f0e', 'marker': 's'},        # Orange, square  
        'int-blasting': {'color': '#2ca02c', 'marker': '^'},   # Green, triangle
        'sls': {'color': '#d62728', 'marker': 'x'},            # Red, x
        '1 core': {'color': '#d62728', 'marker': 'x'},
        '2 cores': {'color': '#ff7f0e', 'marker': 's'},
        '64 cores': {'color': '#2ca02c', 'marker': '^'},
    }
    
    # Additional colors/markers for any extra configurations
    extra_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    extra_markers = ['D', 'v', '<', '>', 'p', '*', 'h', 'H', '|', '_']
    
    result = {
        'colors': [],
        'markers': [],
        'short_names': []
    }
    
    extra_color_idx = 0
    extra_marker_idx = 0
    
    for cfg in configs:
        short_name = get_short_config_name(cfg)
        result['short_names'].append(short_name)
        
        if short_name in style_map:
            result['colors'].append(style_map[short_name]['color'])
            result['markers'].append(style_map[short_name]['marker'])
        else:
            # Use extra colors/markers for unknown configurations
            result['colors'].append(extra_colors[extra_color_idx % len(extra_colors)])
            result['markers'].append(extra_markers[extra_marker_idx % len(extra_markers)])
            extra_color_idx += 1
            extra_marker_idx += 1
    
    return result

def plot_quantile(data, outdir, legend_loc='best', font_size=12, font_family='Gill Sans MT'):
    plt.figure(figsize=(10, 6))
    
    # Set font properties
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family
    })
    
    # Get consistent styling for all configurations
    configs = list(data.keys())
    style_mapping = get_config_style_mapping(configs)
    
    for i, (cfg, results) in enumerate(data.items()):
        # Only include successful runs, exclude timeouts and errors
        times = [v['wall_time'] for v in results.values() if v['status'] == 'SUCCESS' and not np.isnan(v['wall_time'])]
        if times:
            times_sorted = sorted(times)
            xs = np.arange(1, len(times_sorted)+1)
            
            # Determine marker settings based on number of instances
            if len(times_sorted) < 100:
                # For small datasets, show markers for every instance
                marker = style_mapping['markers'][i]
                markersize = 7
                markevery = None  # Show all markers
            else:
                # For large datasets, show markers every N instances to avoid clutter
                marker = style_mapping['markers'][i]
                markersize = 7
                # Show approximately 20-30 markers total
                markevery = max(1, len(times_sorted) // 25)
            
            plt.plot(xs, times_sorted, label=style_mapping['short_names'][i], linewidth=1, 
                    marker=marker, markersize=markersize, markevery=markevery, 
                    color=style_mapping['colors'][i], fillstyle='none')  # Make all markers unfilled
    
    plt.xlabel('Instance solved (sorted by time to solve)', fontsize=font_size, fontfamily=font_family)
    plt.ylabel('Wall time (s)', fontsize=font_size, fontfamily=font_family)
    
    # Apply legend configuration
    if isinstance(legend_loc, tuple):
        plt.legend(bbox_to_anchor=legend_loc, loc='upper left', fontsize=font_size)
    else:
        plt.legend(loc=legend_loc, fontsize=font_size)
    
    plt.grid(True, alpha=0.3)
    
    # Y-axis scale options:
    # plt.yscale('log')  # Full logarithmic
    plt.yscale('symlog', linthresh=1)  # Symmetric log scale with linear threshold at 1
    # plt.yscale('linear')  # Linear scale (default)
    
    # X-axis scale options (uncomment if desired):
    #plt.xscale('log')  # Logarithmic x-axis (focuses on early solved problems)
    # plt.xscale('symlog', linthresh=10)  # Symmetric log x-axis
    
    #plt.title('Quantile Plot (Cactus Plot)', fontsize=font_size + 2, fontfamily=font_family)
    
    # Set tick label font sizes
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    # Ensure absolute path and overwrite existing file
    p = os.path.abspath(os.path.join(outdir, 'quantile.svg'))
    print(f"Saving quantile plot to: {p}")
    plt.savefig(p, format='svg', bbox_inches='tight')
    plt.close()

def plot_scatter(data, outdir, legend_loc='upper left', font_size=12, font_family='Gill Sans MT'):
    # Set font properties
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family
    })
    
    all_times = [v['wall_time'] for results in data.values() for v in results.values() if not np.isnan(v['wall_time'])]
    max_time = PENALTY_TIME_SECONDS if all_times else 0
    
    # Get consistent styling for all configurations
    configs = list(data.keys())
    style_mapping = get_config_style_mapping(configs)
    
    for a, b in itertools.combinations(data.keys(), 2):
        x, y = [], []
        colors = []  # Store color for each point
        markers = []  # Store marker type for each point



        for fn in set(data[a]) & set(data[b]):
            ta = data[a][fn]['wall_time'] if data[a][fn]['status'] == 'SUCCESS' else max_time
            tb = data[b][fn]['wall_time'] if data[b][fn]['status'] == 'SUCCESS' else max_time
            
            # Get the actual result to determine marker type and color
            # Try to get from a first, then b as fallback
            actual_result = data[a][fn].get('actual_result')
            if actual_result not in ['sat', 'unsat']:
                actual_result = data[b][fn].get('actual_result')
                if actual_result not in ['sat', 'unsat']:
                    if data[a][fn]['smt2_stats'].get('expected_status') == 'sat' or data[a][fn]['smt2_stats'].get('expected_status') == 'unsat':
                        actual_result = data[a][fn]['smt2_stats']['expected_status']
                    else:
                        continue
            
            x.append(ta)
            y.append(tb)
            
            # Assign color and marker based on actual result
            if actual_result == 'sat':
                color = 'blue'
                marker = 'o'
            elif actual_result == 'unsat':
                color = 'red'
                marker = 'x'
            else:  # unknown or other
                color = 'gray'
                marker = 's'
            
            markers.append(marker)
            colors.append(color)
        
        plt.figure()
        
        # Group points by type for plotting
        sat_x, sat_y = [], []
        unsat_x, unsat_y = [], []
        other_x, other_y = [], []
        
        for i, (marker, color) in enumerate(zip(markers, colors)):
            if color == 'blue':  # sat
                sat_x.append(x[i])
                sat_y.append(y[i])
            elif color == 'red':  # unsat
                unsat_x.append(x[i])
                unsat_y.append(y[i])
            else:  # unknown/other
                other_x.append(x[i])
                other_y.append(y[i])
        
        # Plot each group with different colors and markers
        if sat_x:
            plt.scatter(sat_x, sat_y, marker='o', alpha=0.45, label='Satisfiable', s=20, facecolors='none', edgecolors='blue')
        if unsat_x:
            plt.scatter(unsat_x, unsat_y, marker='x', c='red', alpha=0.45, label='Unsatisfiable', s=20)
        if other_x:
            plt.scatter(other_x, other_y, marker='s', c='gray', alpha=0.6, label='unknown/other', s=20)
        
        lim = max(max(x, default=0), max(y, default=0))
        plt.plot([0, lim], [0, lim], 'k--', alpha=0.5)
        
        # Get short names for the axis labels and title
        a_short = get_short_config_name(a)
        b_short = get_short_config_name(b)
        
        plt.xlabel(f'{a_short} wall time (s)', fontsize=font_size, fontfamily=font_family)
        plt.ylabel(f'{b_short} wall time (s)', fontsize=font_size, fontfamily=font_family)
        #plt.title(f'Scatter: {a_short} vs {b_short}', fontsize=font_size + 2, fontfamily=font_family)
        plt.grid(True, alpha=0.3)

        # plt.yscale('symlog', linthresh=1)
        # plt.xscale('symlog', linthresh=1)
        
        # Apply legend configuration
        if get_short_config_name(a) == 'polysat' or b_short == 'polysat':
            plt.legend(bbox_to_anchor=(0, 0.92), loc='upper left', fontsize=font_size)
        elif isinstance(legend_loc, tuple):
            plt.legend(bbox_to_anchor=legend_loc, loc='upper left', fontsize=font_size)
        else:
            plt.legend(loc=legend_loc, fontsize=font_size)
        
        # Set tick label font sizes
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        
        # Use short names in filename too
        fname = f'scatter_{a_short.replace(" ","_")}_vs_{b_short.replace(" ","_")}.svg'
        plt.savefig(os.path.join(outdir, fname), format='svg')
        plt.close()

def plot_critical_difference(data, outdir, font_size=12, font_family='Gill Sans MT', file_name='critical_difference'):
    """Generate critical difference plot using Orange's graph_ranks function."""
    # Set font properties
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family
    })

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
    
    # Since we are using ranks, the absolute times don't matter, we just need to
    # ensure that the timed out runs are always ranked last
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
    
    # Use centralized function for consistent short names
    style_mapping = get_config_style_mapping(cfgs)
    shortened_cfgs = style_mapping['short_names']
    
    try:
        cd = compute_CD(mean_ranks, n, alpha='0.05', test='nemenyi')
        output_path = os.path.join(outdir, file_name + '.svg')
        
        print(f"Critical difference threshold: {cd:.4f}")
        print("Pairs with |mean_rank_diff| > CD are significantly different")
        
        # Use Orange's graph_ranks function with adjusted parameters for better layout
        graph_ranks(mean_ranks, shortened_cfgs, cd=cd, 
                   width=4,        # Increase width to give more space
                   textspace=0.82,  # Increase text space for longer names
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

def plot_histogram_smt2_stat(data, outdir, stat_name='total_bit_width', title_suffix='Total Bit-Width', bins=30):
    """
    Plot histogram of SMT2 statistics across all test instances.
    

        Args:
        data: Results data with smt2_stats
        outdir: Output directory for plot  
        stat_name: SMT2 statistic name
        title_suffix: Descriptive name for the statistic
        bins: Number of histogram bins
    """
    plt.figure(figsize=(10, 6))
    
    # Collect all values for the statistic
    stat_values = []
    categories = []
    
    for cfg, results in data.items():
        for test_file, result in results.items():
            if (result['smt2_stats'] is not None and 
                stat_name in result['smt2_stats'] and 
                result['smt2_stats'][stat_name] is not None and
                not np.isnan(result['smt2_stats'][stat_name]) and
                result['smt2_stats'][stat_name] != float('inf')):
                
                stat_values.append(result['smt2_stats'][stat_name])
                categories.append(result['smt2_stats'].get('category', 'unknown'))
    
    if not stat_values:
        print(f"No valid {stat_name} data found for histogram")
        return
    
    # Create histogram
    plt.hist(stat_values, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel(title_suffix)
    plt.ylabel('Number of instances')
    plt.title(f'Distribution of {title_suffix}')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = np.mean(stat_values)
    median_val = np.median(stat_values)
    std_val = np.std(stat_values)
    
    stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}\nCount: {len(stat_values)}'
    plt.text(0.7, 0.7, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    filename = f'histogram_{stat_name}.svg'
    plot_path = os.path.join(outdir, filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved {title_suffix} histogram to: {plot_path}")
    
    # Print category distribution if available
    unique_categories = set(categories)
    if len(unique_categories) > 1 and 'unknown' not in unique_categories:
        print(f"Category distribution:")
        for category in sorted(unique_categories):
            count = categories.count(category)
            print(f"  {category}: {count} instances")

def plot_histogram_stat_binned_performance(data, outdir, stat_name='avg_bit_width', title_suffix='Mean Bit-Width', bins=10, timeout_penalty=PENALTY_TIME_SECONDS, legend_loc=(1.05, 1), font_size=12, font_family='Gill Sans MT'):
    """
    Plot histogram where x-axis bins are based on SMT2 statistics and y-axis shows 
    total solver time per bin. Each bin has multiple bars (one per configuration).
    
    Args:
        data: Results data with smt2_stats
        outdir: Output directory for plot
        stat_name: SMT2 statistic to bin by (e.g., 'avg_bit_width')
        title_suffix: Descriptive name for the statistic
        bins: Number of bins or bin edges
        timeout_penalty: Time in seconds to assign for timeouts/errors (default 5 minutes)
        legend_loc: Legend position ('best', 'upper left', etc. or tuple for bbox_to_anchor)
        font_size: Font size for text elements
        font_family: Font family for text elements
    """
    plt.figure(figsize=(14, 8))
    
    # Set font properties
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family
    })
    
    # Collect all test instances with their statistics and results
    instances_with_stat = []
    configs = list(data.keys())
    
    # Get consistent styling for all configurations
    style_mapping = get_config_style_mapping(configs)
    
    # First pass: collect all instances that have the SMT2 statistic
    all_test_files = set()
    for cfg, results in data.items():
        for test_file in results.keys():
            all_test_files.add(test_file)
    
    for test_file in all_test_files:
        # Find the SMT2 stat value for this test file
        smt2_stat_value = None
        for cfg, results in data.items():
            assert (
                test_file in results and 
                results[test_file]['smt2_stats'] is not None
            )
            if stat_name == 'intops2bitops_ratio' or stat_name == 'vars2ops_ratio':
                intops = 0
                #'bvult', 'bvule', 'bvugt', 'bvuge', 'bvslt', 'bvsle', 'bvsgt', 'bvsge',
                for op in ['bvadd', 'bvsub', 'bvmul', 'bvudiv', 'bvurem', 'bvneg', 'bvsdiv', 'bvsrem', 'bvsmod']:
                    intops += results[test_file]['smt2_stats'][op]
                bitops = 0
                for op in ['bvand', 'bvor', 'bvxor', 'bvnot', 'bvshl', 'bvlshr', 'bvashr', 'bvnand', 'bvnor', 'bvcomp', 'concat']:
                    bitops += results[test_file]['smt2_stats'][op]
                comp = 0
                for op in ['bvult', 'bvule', 'bvugt', 'bvuge', 'bvslt', 'bvsle', 'bvsgt', 'bvsge', '=']:
                    comp += results[test_file]['smt2_stats'][op]
                boolops = 0
                for op in ['and', 'or', 'not', 'xor', '=>', 'ite']:
                    boolops += results[test_file]['smt2_stats'][op]
                totops = intops + bitops + comp + boolops
                if totops > 0:
                    if stat_name == 'vars2ops_ratio':
                        smt2_stat_value = results[test_file]['smt2_stats']['variables'] / totops
                    else:
                        smt2_stat_value = intops / totops
                else:
                    smt2_stat_value = 0
            else:
                assert (
                    stat_name in results[test_file]['smt2_stats'] and 
                    results[test_file]['smt2_stats'][stat_name] is not None and
                    not np.isnan(results[test_file]['smt2_stats'][stat_name]) and
                    results[test_file]['smt2_stats'][stat_name] != float('inf')
                )
                #if results[test_file]['smt2_stats'][stat_name] <= 150.0:
                smt2_stat_value = results[test_file]['smt2_stats'][stat_name]
            break
        
        if smt2_stat_value is not None:
            instance_data = {
                'test_file': test_file,
                'stat_value': smt2_stat_value,
                'cfg_results': {}
            }
            
            # Collect results for each config on this test file
            for cfg, results in data.items():
                if test_file in results:
                    instance_data['cfg_results'][cfg] = results[test_file]
            
            instances_with_stat.append(instance_data)
    
    if not instances_with_stat:
        print(f"No instances found with valid {stat_name} statistics")
        return
    
    print(f"Found {len(instances_with_stat)} instances with {stat_name} data")
    
    # Extract stat values for binning
    stat_values = [instance['stat_value'] for instance in instances_with_stat]
    
    # Create bins
    if isinstance(bins, int) or isinstance(bins, str):
        bin_edges = np.histogram_bin_edges(stat_values, bins=bins, range=(0.1, 0.2))
    else:
        bin_edges = bins
    
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = np.mean(np.diff(bin_edges))
    
    # Assign each instance to a bin
    bin_assignments = np.digitize(stat_values, bin_edges) - 1
    # Handle edge case where max value gets assigned to bin n_bins
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)
    
    # Calculate performance data for each bin and configuration
    n_configs = len(configs)
    
    # Initialize arrays to store results
    bin_successful_times = np.zeros((n_bins, n_configs))
    bin_timeout_times = np.zeros((n_bins, n_configs))
    bin_counts = np.zeros((n_bins, n_configs))
    
    # Populate the arrays
    for i, instance in enumerate(instances_with_stat):
        bin_idx = bin_assignments[i]
        
        for cfg_idx, cfg in enumerate(configs):
            if cfg in instance['cfg_results']:
                result = instance['cfg_results'][cfg]
                bin_counts[bin_idx, cfg_idx] += 1
                
                if result['status'] == 'SUCCESS' and not np.isnan(result['wall_time']):
                    bin_successful_times[bin_idx, cfg_idx] += result['wall_time']
                elif result['status'] in ['TIMEOUT']:
                    bin_timeout_times[bin_idx, cfg_idx] += timeout_penalty
                else:
                    assert False
    
    # Create the plot
    bar_width = bin_width * 0.8 / n_configs  # Width of individual bars
    
    for cfg_idx, cfg in enumerate(configs):
        # Calculate x positions for this config's bars
        x_offset = (cfg_idx - n_configs/2 + 0.5) * bar_width
        x_positions = bin_centers + x_offset
        
        # Use consistent colors and short names from style mapping
        cfg_color = style_mapping['colors'][cfg_idx]
        cfg_short_name = style_mapping['short_names'][cfg_idx]
        
        # Plot successful times (bottom bars)
        plt.bar(x_positions, bin_successful_times[:, cfg_idx], 
                width=bar_width, label=f'{cfg_short_name} (successful)', 
                alpha=0.8, color=cfg_color)
        
        # Plot timeout times on top (stacked)
        plt.bar(x_positions, bin_timeout_times[:, cfg_idx], 
                bottom=bin_successful_times[:, cfg_idx],
                width=bar_width, color=cfg_color, 
                alpha=0.4, hatch='//////', edgecolor='black', linewidth=0.5,
                label=f'{cfg_short_name} (timeout)' if cfg_idx == 0 else "")
    
    plt.xlabel(f'{title_suffix}', fontsize=font_size, fontfamily=font_family)
    plt.ylabel('Total Wall Time (s)', fontsize=font_size, fontfamily=font_family)
    plt.title(f'Solver Performance Binned by {title_suffix}', fontsize=font_size + 2, fontfamily=font_family)
    
    # Apply legend configuration
    if isinstance(legend_loc, tuple):
        plt.legend(bbox_to_anchor=legend_loc, loc='upper left', fontsize=font_size)
    else:
        plt.legend(loc=legend_loc, fontsize=font_size)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.yscale('symlog', linthresh=1)
    
    # Add text annotations with instance counts
    for bin_idx in range(n_bins):
        total_instances = int(np.sum(bin_counts[bin_idx, :]))
        if total_instances > 0:
            # Place text above the tallest bar in this bin
            max_height = np.max(bin_successful_times[bin_idx, :] + bin_timeout_times[bin_idx, :])
            if max_height > 0:
                plt.text(bin_centers[bin_idx], max_height * 1.1, f'{total_instances}', 
                        ha='center', va='bottom', fontsize=font_size-2, rotation=0)
    
    # Set x-axis to show bin edges
    plt.xlim(bin_edges[0] - bin_width*0.5, bin_edges[-1] + bin_width*0.5)
    
    # Add secondary x-axis or improve tick labels to show bin ranges
    ax = plt.gca()
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f'{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}' for i in range(n_bins)], 
                      rotation=45, ha='right', fontsize=font_size)
    
    # Set tick label font sizes
    plt.yticks(fontsize=font_size)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'histogram_binned_performance_{stat_name}.svg'
    plot_path = os.path.join(outdir, filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved binned performance histogram to: {plot_path}")
    
    # Print statistics about the binning
    print(f"\nBinning statistics for {title_suffix}:")
    print(f"Total instances: {len(instances_with_stat)}")
    print(f"Stat value range: {min(stat_values):.2f} to {max(stat_values):.2f}")
    print(f"Number of bins: {n_bins}")
    
    # print(f"\nPer-bin breakdown:")
    # for bin_idx in range(n_bins):
    #     total_instances = int(np.sum(bin_counts[bin_idx, :]))
    #     if total_instances > 0:
    #         bin_range = f"{bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx+1]:.1f}"
    #         print(f"  Bin {bin_range}: {total_instances} instances")
    #         for cfg_idx, cfg in enumerate(configs):
    #             if bin_counts[bin_idx, cfg_idx] > 0:
    #                 successful_time = bin_successful_times[bin_idx, cfg_idx]
    #                 timeout_time = bin_timeout_times[bin_idx, cfg_idx]
    #                 n_timeouts = int(timeout_time / timeout_penalty) if timeout_penalty > 0 else 0
    #                 n_instances = int(bin_counts[bin_idx, cfg_idx])
    #                 n_successful = n_instances - n_timeouts
    #                 print(f"    {cfg}: {n_instances} instances, {n_successful} successful ({successful_time:.1f}s), "
    #                       f"{n_timeouts} timeout/error ({timeout_time:.1f}s)")

def plot_histogram_family_binned_performance(data, outdir, timeout_penalty=PENALTY_TIME_SECONDS, legend_loc=(1.05, 1), font_size=12, font_family='Gill Sans MT'):
    """
    Plot histogram where x-axis bins are based on benchmark families and y-axis shows 
    total solver time per family. Each family has multiple bars (one per configuration).
    
    Args:
        data: Results data with smt2_stats
        outdir: Output directory for plot
        timeout_penalty: Time in seconds to assign for timeouts/errors (default 5 minutes)
        legend_loc: Legend position ('best', 'upper left', etc. or tuple for bbox_to_anchor)
        font_size: Font size for text elements
        font_family: Font family for text elements
    """
    plt.figure(figsize=(16, 8))
    
    # Set font properties
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family
    })
    
    # Extract family information from test file paths
    family_instances = {}
    configs = list(data.keys())
    
    # Get consistent styling for all configurations
    style_mapping = get_config_style_mapping(configs)
    
    # First pass: collect all instances and extract their families
    all_test_files = set()
    for cfg, results in data.items():
        for test_file in results.keys():
            all_test_files.add(test_file)
    
    for test_file in all_test_files:
        # Extract family from the test file path
        path_parts = test_file.split('/')
        family = None
        
        # Look for the family name after SMT-COMP_2024 or other dataset indicators
        for i, part in enumerate(path_parts):
            if 'SMT-COMP' in part or 'VLSAT' in part or any(dataset in part for dataset in ['sage', 'wintersteiger']):
                if i + 1 < len(path_parts):
                    family = path_parts[i + 1]
                    break
        
        # Alternative approach: look for common SMT-LIB family patterns
        if family is None:
            # Try extracting from common SMT-LIB benchmark structure
            # Pattern could be: QF_BV/family_name/file.smt2 or family_name/subfolder/file.smt2
            for i, part in enumerate(path_parts):
                if part.startswith('QF_'):
                    # Logic name found, next part might be family
                    if i + 1 < len(path_parts):
                        family = path_parts[i + 1]
                        break
        
        # Another approach: if the path has SMT-COMP in it, look for specific patterns
        if family is None and any('SMT-COMP' in part for part in path_parts):
            # For SMT-COMP, families are often direct subdirectories
            # Look for path patterns like: SMT-COMP_2024/QF_BV/family/file.smt2
            smt_comp_idx = None
            for i, part in enumerate(path_parts):
                if 'SMT-COMP' in part:
                    smt_comp_idx = i
                    break
            
            if smt_comp_idx is not None:
                # Skip SMT-COMP_YEAR and potentially QF_BV logic
                potential_family_indices = []
                for j in range(smt_comp_idx + 1, len(path_parts) - 1):  # -1 to skip filename
                    part = path_parts[j]
                    # Skip logic names
                    if not part.startswith('QF_') and part not in ['incremental', 'non-incremental']:
                        potential_family_indices.append(j)
                
                if potential_family_indices:
                    family_idx = potential_family_indices[0]  # Take first non-logic directory
                    family = path_parts[family_idx]
        
        # Fallback: use the first directory component
        if family is None and len(path_parts) > 1:
            family = path_parts[0]
        
        # If still no family, use a default
        if family is None:
            family = 'unknown'
        
        # Clean up family name (remove common suffixes/prefixes and numbers with dashes)
        family = family.split('.')[0]  # Remove file extensions if present
        
        # Remove numbers and dash at the beginning (e.g., "01-family" -> "family")
        import re
        family = re.sub(r'^\d+-', '', family)
        
        if family not in family_instances:
            family_instances[family] = []
        
        # Collect results for each config on this test file
        instance_data = {
            'test_file': test_file,
            'cfg_results': {}
        }
        
        for cfg, results in data.items():
            if test_file in results:
                instance_data['cfg_results'][cfg] = results[test_file]
        
        family_instances[family].append(instance_data)
    
    if not family_instances:
        print("No benchmark families found")
        return
    
    # Calculate performance data for each family and configuration
    n_configs = len(configs)
    
    # Calculate total times for filtering
    family_total_times = {}
    for family, instances in family_instances.items():
        family_successful_times = [0] * n_configs
        family_timeout_times = [0] * n_configs
        
        for instance in instances:
            for cfg_idx, cfg in enumerate(configs):
                if cfg in instance['cfg_results']:
                    result = instance['cfg_results'][cfg]
                    
                    if result['status'] == 'SUCCESS' and not np.isnan(result['wall_time']):
                        family_successful_times[cfg_idx] += result['wall_time']
                    elif result['status'] in ['TIMEOUT']:
                        family_timeout_times[cfg_idx] += timeout_penalty
        
        # Calculate total time for each config (successful + timeout)
        total_times = [family_successful_times[i] + family_timeout_times[i] for i in range(n_configs)]
        family_total_times[family] = total_times
    
    # Filter out families where any solver takes less than 5 seconds total
    MIN_TOTAL_TIME = 5.0  # seconds
    filtered_families = []
    for family, total_times in family_total_times.items():
        if all(time >= MIN_TOTAL_TIME for time in total_times):
            filtered_families.append(family)
        else:
            print(f"Filtering out family '{family}' - min total time: {min(total_times):.2f}s")
    
    # Update family_instances to only include filtered families
    family_instances = {family: family_instances[family] for family in filtered_families}
    
    # Sort families by name for consistent ordering
    families = sorted(family_instances.keys())
    n_families = len(families)
    
    print(f"Found {n_families} benchmark families after filtering:")
    for family in families:
        print(f"  {family}: {len(family_instances[family])} instances")
    
    # Initialize arrays to store results
    family_successful_times = np.zeros((n_families, n_configs))
    family_timeout_times = np.zeros((n_families, n_configs))
    family_counts = np.zeros((n_families, n_configs))
    
    # Populate the arrays
    for family_idx, family in enumerate(families):
        for instance in family_instances[family]:
            for cfg_idx, cfg in enumerate(configs):
                if cfg in instance['cfg_results']:
                    result = instance['cfg_results'][cfg]
                    family_counts[family_idx, cfg_idx] += 1
                    
                    if result['status'] == 'SUCCESS' and not np.isnan(result['wall_time']):
                        family_successful_times[family_idx, cfg_idx] += result['wall_time']
                    elif result['status'] in ['TIMEOUT']:
                        family_timeout_times[family_idx, cfg_idx] += timeout_penalty
                    else:
                        assert False

    # Create the plot
    x_positions = np.arange(n_families)
    bar_width = 0.8 / n_configs  # Width of individual bars
    
    for cfg_idx, cfg in enumerate(configs):
        # Calculate x positions for this config's bars
        x_offset = (cfg_idx - n_configs/2 + 0.5) * bar_width
        x_pos = x_positions + x_offset
        
        # Use consistent colors and short names from style mapping
        cfg_color = style_mapping['colors'][cfg_idx]
        cfg_short_name = style_mapping['short_names'][cfg_idx]
        
        # Plot successful times (bottom bars)
        plt.bar(x_pos, family_successful_times[:, cfg_idx], 
                width=bar_width, label=f'{cfg_short_name} (successful)', 
                alpha=0.8, color=cfg_color)
        
        # Plot timeout times on top (stacked) - create separate legend entry for each config
        plt.bar(x_pos, family_timeout_times[:, cfg_idx], 
                bottom=family_successful_times[:, cfg_idx],
                width=bar_width, color=cfg_color, 
                alpha=0.4, hatch='//////', edgecolor='black', linewidth=0.5,
                label=f'{cfg_short_name} (timeout)')
    
    #plt.xlabel('Benchmark Family', fontsize=font_size, fontfamily=font_family)
    plt.ylabel('Total wall time (s)', fontsize=font_size, fontfamily=font_family)
    #plt.title('Solver Performance by Benchmark Family', fontsize=font_size + 2, fontfamily=font_family)
    
    # Apply legend configuration with horizontal layout
    # Calculate number of columns for horizontal legend (2 entries per config: successful + timeout)
    n_legend_entries = n_configs * 2
    #ncol = min(n_legend_entries, 4)  # Limit to max 4 columns for readability
    ncol = 4

    if isinstance(legend_loc, tuple):
        plt.legend(bbox_to_anchor=legend_loc, fontsize=font_size, ncol=ncol)
    else:
        plt.legend(loc=legend_loc, fontsize=font_size, ncol=ncol)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.yscale('symlog', linthresh=1)
    
    # Add text annotations with instance counts
    for family_idx in range(n_families):
        total_instances = int(np.sum(family_counts[family_idx, :]))
        if total_instances > 0:
            # Place text above the tallest bar in this family
            max_height = np.max(family_successful_times[family_idx, :] + family_timeout_times[family_idx, :])
            if max_height > 0:
                plt.text(x_positions[family_idx], max_height * 1.1, f'{total_instances//4}', 
                        ha='center', va='bottom', fontsize=font_size-2, rotation=0)
    
    # Set x-axis labels with font size
    plt.xticks(x_positions, families, rotation=45, ha='right', fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'histogram_family_binned_performance.svg'
    plot_path = os.path.join(outdir, filename)
    plt.savefig(plot_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved family-binned performance histogram to: {plot_path}")

def rename_dict_keys(data, old_to_new_names):
    return {old_to_new_names.get(k, k): v for k, v in data.items()}

def sort_dict_keys(data):
    return {k: v for k, v in sorted(list(data.items()))}

def main():
    global DATASETS_FOLDER

    OPT = "SMT-COMP"
    # OPT = "VLSAT3g"
    # OPT = "VLSAT3a"
    # OPT = "smart-contracts"
    #OPT = "parallel-scaling"

    if OPT == "SMT-COMP" or OPT == "parallel-scaling":
        DATASETS_FOLDER = 'D:/datasets/'
    else:
        DATASETS_FOLDER = 'datasets/'

    input_names = None
    input1 = None
    input2 = None
    input3 = None
    input4 = None
    input5 = None
    if OPT == "SMT-COMP":
        input1 = "20250612_142520_smt-comp_2024"
        input2 = "20250618_151400_smt-comp_2024"
        input3 = "20250621_045934_smt-comp_2024-electric-boogaloo"
        input_names = [input1, input2, input3]
    elif OPT == "VLSAT3a":
        input1 = '20250620_134236_vlsat3_a_rest'
        input2 = '20250612_144947_vlsat3_a'
        input_names = [input1, input2]
    elif OPT == "VLSAT3g":
        input1 = '20250612_144947_vlsat3_g'
    elif OPT == "smart-contracts":
        input1 = '20250612_140513_smart_contracts'
    elif OPT == "parallel-scaling":
        input1 = "20250612_142520_smt-comp_2024"
        input2 = "20250618_151400_smt-comp_2024"
        input3 = "20250621_045934_smt-comp_2024-electric-boogaloo"
        input4 = "20250621_120126_parallel-scaling-64"
        input5 = "20250621_155102_parallel-scaling-2"
        input_names = [input1, input2, input3, input4, input5]
    else:
        assert False
    
    if OPT == "VLSAT3g" or OPT == "smart-contracts":
        input_names = [input1]

    cache_file_path = get_cleaned_data_cache_path(input_names)

    data = load_cleaned_data_cache(cache_file_path)
    #data = None
    if data is not None:
        print(f"\n=== USING CACHED DATA ===")
        print("Skipping parsing, merging, and cleaning - using cached results")
    else:
        print(f"\n=== NO CACHE FOUND, PROCESSING DATA ===")
        
        # Parse multiple SMT-COMP 2024 datasets for merging
        datasets = []

        if input4 is not None:
            print(f"Parsing dataset 4: {input4}")
            data4 = parse_results_benchy(Path(RESULTS_FOLDER) / input4)
            datasets.append(rename_dict_keys(data4, {'z3-parallel': '64 cores'}))
        
        if input5 is not None:
            print(f"Parsing dataset 5: {input5}")
            data5 = parse_results_benchy(Path(RESULTS_FOLDER) / input5)
            datasets.append(rename_dict_keys(data5, {'z3-parallel': '2 cores'}))
            
        print(f"Parsing dataset 1: {input1}")
        data1 = parse_results_benchy(Path(RESULTS_FOLDER) / input1)
        if(OPT == 'parallel-scaling'):
            data1.pop('z3-int-blast')
            data1.pop('z3-lazy-bit-blast')
            data1.pop('z3-sls-and-bit-blasting-sequential')
            data1 = rename_dict_keys(data1, {'z3-bit-blast': '1 core'})
        datasets.append(data1)
        
        if input2 is not None:
            print(f"Parsing dataset 2: {input2}")
            data2 = parse_results_benchy(Path(RESULTS_FOLDER) / input2)
            if(OPT == 'parallel-scaling'):
                data2.pop('z3-int-blast')
                data2.pop('z3-lazy-bit-blast')
                data2.pop('z3-sls-and-bit-blasting-sequential')
                data2 = rename_dict_keys(data2, {'z3-bit-blast': '1 core'})
            datasets.append(data2)

        if input3 is not None:
            print(f"Parsing dataset 3: {input3}")
            data3 = parse_results_benchy(Path(RESULTS_FOLDER) / input3)
            if(OPT == 'parallel-scaling'):
                data3.pop('z3-int-blast')
                data3.pop('z3-lazy-bit-blast')
                data3.pop('z3-sls-and-bit-blasting-sequential')
                data3 = rename_dict_keys(data3, {'z3-bit-blast': '1 core'})
            datasets.append(data3)
        
        
        # Merge datasets intelligently
        print(f"\n=== MERGING SMT-COMP 2024 DATASETS ===")
        data, merge_stats = merge_and_clean_multiple_datasets(datasets, includeTimeout=INCLUDE_TIMEOUT)

        data = sort_dict_keys(data)

        # Save the cleaned data to cache for future runs
        save_cleaned_data_cache(data, cache_file_path)
    
    data = sort_dict_keys(data)

    # Set output directory based on merged data
    if OPT == "parallel-scaling":
        output_directory = Path(OUTPUT) / f"parallel-scaling"
    else:
        output_directory = Path(OUTPUT) / f"merged_{input1}_{input2}"
    os.makedirs(output_directory, exist_ok=True)

    # Filter the task file to remove instances solved by all configs
    # task_file = Path("SMT-COMP_2024_tasks_really_all.txt")
    # filtered_task_file = Path(f"SMT-COMP_2024_tasks_challenging_merged.txt")
    
    # if task_file.exists():
    #     filter_task_file_remove_solved(task_file, data, filtered_task_file, keep_solved=False)
    # else:
    #     print(f"Task file {task_file} not found, skipping task file filtering")
    
    # Generate plots with configurable legend positions and font settings
    if OPT == "VLSAT3a":
        plot_quantile(data, output_directory, legend_loc='lower right', font_size=16, font_family='Gill Sans MT')
    elif OPT == "smart-contracts":
        plot_quantile(data, output_directory, legend_loc='lower right', font_size=22, font_family='Gill Sans MT')
    else:
        plot_quantile(data, output_directory, legend_loc='best', font_size=16, font_family='Gill Sans MT')

    plot_critical_difference(data, output_directory, font_size=12, font_family='Gill Sans MT')

    if OPT == "SMT-COMP":
        family_instances = {}

        # First pass: collect all instances and extract their families
        all_test_files = set()
        for cfg, results in data.items():
            for test_file in results.keys():
                all_test_files.add(test_file)
        for test_file in all_test_files:
            # Extract family from the test file path
            path_parts = test_file.split('/')
            family = None
            
            # Look for the family name after SMT-COMP_2024 or other dataset indicators
            for i, part in enumerate(path_parts):
                if 'SMT-COMP' in part or 'VLSAT' in part or any(dataset in part for dataset in ['sage', 'wintersteiger']):
                    if i + 1 < len(path_parts):
                        family = path_parts[i + 1]
                        break
            
            # Alternative approach: look for common SMT-LIB family patterns
            if family is None:
                # Try extracting from common SMT-LIB benchmark structure
                # Pattern could be: QF_BV/family_name/file.smt2 or family_name/subfolder/file.smt2
                for i, part in enumerate(path_parts):
                    if part.startswith('QF_'):
                        # Logic name found, next part might be family
                        if i + 1 < len(path_parts):
                            family = path_parts[i + 1]
                            break
            
            # Another approach: if the path has SMT-COMP in it, look for specific patterns
            if family is None and any('SMT-COMP' in part for part in path_parts):
                # For SMT-COMP, families are often direct subdirectories
                # Look for path patterns like: SMT-COMP_2024/QF_BV/family/file.smt2
                smt_comp_idx = None
                for i, part in enumerate(path_parts):
                    if 'SMT-COMP' in part:
                        smt_comp_idx = i
                        break
                
                if smt_comp_idx is not None:
                    # Skip SMT-COMP_YEAR and potentially QF_BV logic
                    potential_family_indices = []
                    for j in range(smt_comp_idx + 1, len(path_parts) - 1):  # -1 to skip filename
                        part = path_parts[j]
                        # Skip logic names
                        if not part.startswith('QF_') and part not in ['incremental', 'non-incremental']:
                            potential_family_indices.append(j)
                    
                    if potential_family_indices:
                        family_idx = potential_family_indices[0]  # Take first non-logic directory
                        family = path_parts[family_idx]
            
            # Fallback: use the first directory component
            if family is None and len(path_parts) > 1:
                family = path_parts[0]
            
            # If still no family, use a default
            if family is None:
                family = 'unknown'
            
            # Clean up family name (remove common suffixes/prefixes and numbers with dashes)
            family = family.split('.')[0]  # Remove file extensions if present
            
            # Remove numbers and dash at the beginning (e.g., "01-family" -> "family")
            # import re
            # family = re.sub(r'^\d+-', '', family)
            
            if family not in family_instances:
                family_instances[family] = []
            
            # Collect results for each config on this test file
            instance_data = {
                'test_file': test_file,
                'cfg_results': {}
            }
            
            for cfg, results in data.items():
                if test_file in results:
                    instance_data['cfg_results'][cfg] = results[test_file]
            
            family_instances[family].append(instance_data)
        
        if not family_instances:
            print("No benchmark families found")
            return

        for family, _ in family_instances.items():
            print(f"  {family}: {len(family_instances[family])} instances")
            data_family = {}
            for cfg, results in data.items():
                data_family[cfg] = {}
                for test_case, result in results.items():
                    if family in test_case:
                        data_family[cfg][test_case] = result
            plot_critical_difference(data_family, output_directory, font_size=12, font_family='Gill Sans MT', file_name=family)
            

    if OPT != "parallel-scaling":
        plot_scatter(data, output_directory, legend_loc='upper left', font_size=12, font_family='Gill Sans MT')


        # plot_histogram_smt2_stat(data, output_directory, stat_name='avg_bit_width', title_suffix='Mean Bit-Width')

        # plot_histogram_smt2_stat(data, output_directory, stat_name='variables', title_suffix='vars')

        # plot_histogram_stat_binned_performance(data, output_directory, 
        #     stat_name='avg_bit_width', title_suffix='Mean Bit-Width', legend_loc=(1.05, 1), 
        #     font_size=12, font_family='Gill Sans MT')
        # plot_histogram_stat_binned_performance(data, output_directory,
        #     stat_name='intops2bitops_ratio', title_suffix='Intops to total ratio',
        #     bins='doane', legend_loc=(1.05, 1), font_size=12, font_family='Gill Sans MT')
        # plot_histogram_stat_binned_performance(data, output_directory,
        #     stat_name='vars2ops_ratio', title_suffix='Vars to ops ratio',
        #     bins='doane', legend_loc=(1.05, 1), font_size=12, font_family='Gill Sans MT')

        plot_histogram_family_binned_performance(data, output_directory, legend_loc='upper right', font_size=16, font_family='Gill Sans MT')
        # plot_histogram_stat_binned_performance(data, output_directory, stat_name='var_to_ops_ratio', title_suffix='Variable to Operations Ratio')
        # plot_histogram_stat_binned_performance(data, output_directory, stat_name='bool_to_bitvec_ratio', title_suffix='Boolean to Bitvector Ratio')


if __name__ == '__main__':
    main()
