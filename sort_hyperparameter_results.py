#!/usr/bin/env python3
"""
Sort and analyze hyperparameter search results from rank data.
"""

import argparse
import re
from pathlib import Path


def parse_ranks_file(file_path):
    """
    Parse a ranks file and extract configuration names and their mean ranks.
    
    Args:
        file_path: Path to the ranks file
    
    Returns:
        list: List of tuples (config_name, rank)
    """
    ranks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Look for lines with pattern "config_name: rank_value"
            match = re.match(r'^\s*([^:]+):\s*(\d+\.?\d*)\s*$', line)
            if match:
                config_name = match.group(1).strip()
                rank_value = float(match.group(2))
                ranks.append((config_name, rank_value))
    
    return ranks


def analyze_configs(ranks):
    """
    Analyze configuration patterns and extract insights.
    
    Args:
        ranks: List of tuples (config_name, rank)
    
    Returns:
        dict: Analysis results
    """
    analysis = {
        'best_configs': [],
        'worst_configs': [],
        'by_algorithm': {},
        'by_parameter': {}
    }
    
    # Sort by rank (lower is better)
    sorted_ranks = sorted(ranks, key=lambda x: x[1])
    
    # Best and worst configurations
    analysis['best_configs'] = sorted_ranks[:10]  # Top 10
    analysis['worst_configs'] = sorted_ranks[-10:]  # Bottom 10
    
    # Group by algorithm type
    for config_name, rank in ranks:
        # Extract algorithm type (first part before underscore or space)
        if config_name.startswith('CaC'):
            algo_type = 'CaC'
        elif config_name.startswith('DDFW'):
            algo_type = 'DDFW'
        elif config_name.startswith('SLS'):
            algo_type = 'SLS'
        elif config_name.startswith('SAT'):
            algo_type = 'SAT'
        else:
            algo_type = 'Other'
        
        if algo_type not in analysis['by_algorithm']:
            analysis['by_algorithm'][algo_type] = []
        analysis['by_algorithm'][algo_type].append((config_name, rank))
    
    # Sort each algorithm group
    for algo_type in analysis['by_algorithm']:
        analysis['by_algorithm'][algo_type].sort(key=lambda x: x[1])
    
    return analysis


def extract_parameters(config_name):
    """
    Extract parameters from configuration name.
    
    Args:
        config_name: Configuration name string
    
    Returns:
        dict: Extracted parameters
    """
    params = {}
    
    # Look for common parameter patterns
    param_patterns = [
        (r'bs(\d+)', 'batch_size'),
        (r'd(\d+)', 'depth'),
        (r'r(\d+)', 'restart'),
        (r'bf(\d+)', 'backtrack_factor'),
        (r'CaC(\d+)', 'cac_variant'),
        (r'DDFW(\d+)', 'ddfw_variant'),
        (r'SLS(\d+)', 'sls_variant'),
        (r'SAT(\d+)', 'sat_variant')
    ]
    
    for pattern, param_name in param_patterns:
        match = re.search(pattern, config_name)
        if match:
            params[param_name] = int(match.group(1))
    
    return params


def print_analysis(analysis):
    """
    Print detailed analysis results.
    
    Args:
        analysis: Analysis results dictionary
    """
    print("=== HYPERPARAMETER SEARCH RESULTS ANALYSIS ===\n")
    
    # Best configurations
    print("🏆 TOP 10 BEST CONFIGURATIONS (Lower rank = better):")
    print("-" * 60)
    for i, (config_name, rank) in enumerate(analysis['best_configs'], 1):
        print(f"{i:2d}. {config_name:<50} {rank:7.2f}")
    
    print("\n" + "="*70 + "\n")
    
    # Worst configurations
    print("💥 TOP 10 WORST CONFIGURATIONS:")
    print("-" * 60)
    for i, (config_name, rank) in enumerate(analysis['worst_configs'], 1):
        print(f"{i:2d}. {config_name:<50} {rank:7.2f}")
    
    print("\n" + "="*70 + "\n")
    
    # Algorithm type analysis
    print("📊 ANALYSIS BY ALGORITHM TYPE:")
    print("-" * 60)
    
    for algo_type, configs in analysis['by_algorithm'].items():
        if not configs:
            continue
            
        best_rank = min(rank for _, rank in configs)
        worst_rank = max(rank for _, rank in configs)
        avg_rank = sum(rank for _, rank in configs) / len(configs)
        
        print(f"\n{algo_type} Algorithm:")
        print(f"  Configurations: {len(configs)}")
        print(f"  Best rank:      {best_rank:.2f}")
        print(f"  Worst rank:     {worst_rank:.2f}")
        print(f"  Average rank:   {avg_rank:.2f}")
        
        # Show top 3 for this algorithm
        print(f"  Top 3 {algo_type} configs:")
        for i, (config_name, rank) in enumerate(configs[:3], 1):
            short_name = config_name.replace(f'{algo_type}_', '').replace(f'{algo_type}', '')
            print(f"    {i}. {short_name:<35} {rank:7.2f}")


def save_sorted_results(ranks, output_file):
    """
    Save sorted results to a file.
    
    Args:
        ranks: List of tuples (config_name, rank)
        output_file: Output file path
    """
    sorted_ranks = sorted(ranks, key=lambda x: x[1])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== SORTED HYPERPARAMETER SEARCH RESULTS ===\n")
        f.write("Rank | Configuration Name\n")
        f.write("-" * 80 + "\n")
        
        for i, (config_name, rank) in enumerate(sorted_ranks, 1):
            f.write(f"{rank:7.2f} | {config_name}\n")
        
        f.write(f"\nTotal configurations: {len(sorted_ranks)}\n")
        f.write(f"Best rank: {sorted_ranks[0][1]:.2f} ({sorted_ranks[0][0]})\n")
        f.write(f"Worst rank: {sorted_ranks[-1][1]:.2f} ({sorted_ranks[-1][0]})\n")


def main():
    parser = argparse.ArgumentParser(
        description='Sort and analyze hyperparameter search results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sort_hyperparameter_results.py parallel_hyperparameter_search_ranks.txt
  python sort_hyperparameter_results.py ranks.txt --output sorted_ranks.txt
  python sort_hyperparameter_results.py ranks.txt --top 20
        """
    )
    
    parser.add_argument('input_file', type=str, help='Path to the input ranks file')
    parser.add_argument('--output', type=str, help='Path to save sorted results (optional)')
    parser.add_argument('--top', type=int, default=10, help='Number of top/bottom configs to show (default: 10)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return 1
    
    print(f"Reading ranks from: {input_file}")
    
    # Parse the ranks file
    ranks = parse_ranks_file(input_file)
    
    if not ranks:
        print("Error: No valid rank data found in input file")
        return 1
    
    print(f"Found {len(ranks)} configurations")
    
    # Analyze the results
    analysis = analyze_configs(ranks)
    
    # Update top/bottom counts based on user input
    analysis['best_configs'] = analysis['best_configs'][:args.top]
    analysis['worst_configs'] = analysis['worst_configs'][-args.top:]
    
    # Print analysis
    print_analysis(analysis)
    
    # Save results if output file specified
    if args.output:
        output_file = Path(args.output)
        save_sorted_results(ranks, output_file)
        print(f"\n📁 Sorted results saved to: {output_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())