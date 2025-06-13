mkdir -p ./datasets/VLSAT3 \
    && wget -r --no-parent -A "vlsat3_a*.smt2.bz2" https://cadp.inria.fr/ftp/benchmarks/vlsat/ -P ./datasets/VLSAT3 \
    && find ./datasets/VLSAT3 -name "*.smt2.bz2" -exec bunzip2 {} +
