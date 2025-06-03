#!/usr/bin/env bash
set -eo pipefail

# Run SMT-COMP QF_BV Benchmark
echo "=== Running SMT-COMP QF_BV Benchmark ==="
benchexec smtcomp-qfbv-benchmark.xml --limitCores 1 --no-hyperthreading --numOfThreads 1

echo "--- Generating tables for SMT-COMP QF_BV ---"
table-generator results/smtcomp-qfbv-benchmark.*.bz2
# Generate diff-only table
table-generator --only-differences results/smtcomp-qfbv-benchmark.*.bz2

# Print generated HTML links
echo "SMT-COMP HTML:"
ls results/smtcomp-qfbv-benchmark.*.html
ls results/smtcomp-qfbv-benchmark.*.html | sed 's/\.html$/.table.html/' || true

# Run VLSAT3 QF_BV Benchmark

echo "=== Running VLSAT3 QF_BV Benchmark ==="
benchexec vlsat-qfbv-benchmark.xml --limitCores 1 --no-hyperthreading --numOfThreads 1

echo "--- Generating tables for VLSAT3 QF_BV ---"
table-generator results/vlsat-qfbv-benchmark.*.bz2
# Generate diff-only table
table-generator --only-differences results/vlsat-qfbv-benchmark.*.bz2

# Print generated HTML links
echo "VLSAT3 HTML:"
ls results/vlsat-qfbv-benchmark.*.html
ls results/vlsat-qfbv-benchmark.*.html | sed 's/\.html$/.table.html/' || true
