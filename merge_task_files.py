#!/usr/bin/env python3
"""
Take everything from file1 and remove everything that appears in file2 (set difference).
"""

import argparse
from pathlib import Path


def merge_files(file1_path, file2_path, output_path, preserve_order=False):
    """
    Take everything from file1 and remove everything that appears in file2 (set difference).
    
    Args:
        file1_path: Path to the first input file (keep these)
        file2_path: Path to the second input file (remove these)
        output_path: Path to the output file
        preserve_order: If True, preserve order from file1. If False, sort alphabetically.
    
    Returns:
        tuple: (total_lines_file1, total_lines_file2, remaining_lines, removed_lines)
    """
    
    # Read first file
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            lines_file1 = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {file1_path}: {e}")
        return 0, 0, 0, 0
    
    # Read second file
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            lines_file2 = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file {file2_path}: {e}")
        return len(lines_file1), 0, 0, 0
    
    print(f"File 1: {len(lines_file1)} lines")
    print(f"File 2: {len(lines_file2)} lines")
    
    # Convert to sets for difference calculation
    set1 = set(lines_file1)
    set2 = set(lines_file2)
    
    # Find difference (lines in file1 but not in file2)
    difference = set1 - set2
    
    # Find lines that were removed
    removed = set1 & set2
    
    if preserve_order:
        # Preserve order from file1 for remaining lines
        remaining_lines = [line for line in lines_file1 if line in difference]
    else:
        # Sort alphabetically
        remaining_lines = sorted(difference)
    
    print(f"Lines from file 1 kept: {len(remaining_lines)}")
    print(f"Lines from file 1 removed (found in file 2): {len(removed)}")
    print(f"Lines only in file 2 (ignored): {len(set2 - set1)}")
    
    # Write difference to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in remaining_lines:
                f.write(line + '\n')
        print(f"Difference file written to: {output_path}")
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")
        return len(lines_file1), len(lines_file2), len(remaining_lines), len(removed)
    
    return len(lines_file1), len(lines_file2), len(remaining_lines), len(removed)


def main():
    parser = argparse.ArgumentParser(
        description='Take everything from file1 and remove everything that appears in file2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python merge_task_files.py file1.txt file2.txt difference.txt
  python merge_task_files.py file1.txt file2.txt difference.txt --preserve-order
  python merge_task_files.py SMT-COMP_2024_tasks_all.txt SMT-COMP_2024_tasks_solved.txt SMT-COMP_2024_tasks_remaining.txt
        """
    )
    
    parser.add_argument('file1', type=str, help='Path to the first input file (keep these lines)')
    parser.add_argument('file2', type=str, help='Path to the second input file (remove these lines)')
    parser.add_argument('output', type=str, help='Path to the output file (file1 - file2)')
    parser.add_argument('--preserve-order', action='store_true', 
                        help='Preserve order from file1 (default: sort alphabetically)')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    file1_path = Path(args.file1)
    file2_path = Path(args.file2)
    output_path = Path(args.output)
    
    # Check input files exist
    if not file1_path.exists():
        print(f"Error: File {file1_path} does not exist")
        return 1
    
    if not file2_path.exists():
        print(f"Error: File {file2_path} does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Computing set difference (file1 - file2):")
    print(f"  File 1 (keep): {file1_path}")
    print(f"  File 2 (remove): {file2_path}")
    print(f"  Output: {output_path}")
    print(f"  Order: {'preserved from file1' if args.preserve_order else 'alphabetically sorted'}")
    print()
    
    # Perform the set difference
    file1_count, file2_count, remaining_count, removed_count = merge_files(
        file1_path, file2_path, output_path, args.preserve_order
    )
    
    if remaining_count >= 0:
        print(f"\n✓ Successfully computed set difference!")
        print(f"  File 1: {file1_count} lines")
        print(f"  File 2: {file2_count} lines") 
        print(f"  Remaining (file1 - file2): {remaining_count} lines")
        print(f"  Removed from file1: {removed_count} lines")
        print(f"  Retention rate: {remaining_count/file1_count*100:.1f}%")
        return 0
    else:
        print(f"\n✗ Failed to compute set difference")
        return 1


if __name__ == '__main__':
    exit(main())