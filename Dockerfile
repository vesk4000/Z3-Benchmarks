# Use Ubuntu 24.04 LTS as the base image
FROM ubuntu:24.04

# Set non-interactive mode to avoid prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install ALL the tools you might need for development and benchmarking
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl python3 python3-pip python3-venv \
    vim nano emacs less htop tree unzip zip \
    ca-certificates procps lsb-release software-properties-common \
    gdb valgrind strace ltrace \
    tmux screen \
    openssh-client \
    jq \
    time \
    zstd bzip2 wget \
    sudo fuse-overlayfs \
    && rm -rf /var/lib/apt/lists/*

# Add the BenchExec PPA and install BenchExec (as recommended in the guide)
RUN add-apt-repository ppa:sosy-lab/benchmarking \
    && apt-get update \
    && apt-get install -y benchexec \
    && rm -rf /var/lib/apt/lists/*

# Create directories for multiple Z3 versions
RUN mkdir -p /opt/z3-versions

# Build Z3 4.12.0
# RUN git clone https://github.com/Z3Prover/z3.git z3-4.12.0 \
#     && cd z3-4.12.0 \
#     && git checkout z3-4.12.0 \
#     && python3 scripts/mk_make.py --prefix=/opt/z3-versions/z3-4.12.0/install \
#     && cd build && make -j"$(nproc)" install \
#     && rm -rf build .git

# Build Z3 latest (main branch)
RUN git clone https://github.com/Z3Prover/z3.git z3-4.15.1 \
    && cd z3-4.15.1 \
    && python3 scripts/mk_make.py --prefix=/opt/z3-versions/z3-4.15.1/install \
    && cd build && make -j"$(nproc)" install \
    && rm -rf build .git

# Build Z3 PolySAT
RUN git clone https://github.com/Z3Prover/z3.git z3-poly \
    && cd z3-poly \
    && git checkout 16fb86b636047fd79ad5827f768b6f26d8812948 \
    && python3 scripts/mk_make.py --prefix=/opt/z3-versions/z3-poly/install \
    && cd build && make -j"$(nproc)" install \
    && rm -rf build .git

# Create symlinks for easy access - default to latest
RUN ln -s /opt/z3-versions/z3-4.15.1/install/bin/z3 /usr/local/bin/z3 \
    #&& ln -s /opt/z3-versions/z3-poly/install/bin/z3 /usr/local/bin/z3-poly \ 
    # && ln -s /opt/z3-versions/z3-4.12.0/install/bin/z3 /usr/local/bin/z3-4.12.0 \
    && ln -s /opt/z3-versions/z3-4.15.1/install/bin/z3 /usr/local/bin/z3-4.15.1

# Install Python packages that might be useful for benchmarking
RUN pip3 install --break-system-packages \
    pandas matplotlib seaborn \
    jupyter notebook \
    pyyaml \
    requests

# Create workspace and benchmark directories
RUN mkdir -p /workspace /workspace/benchmarks /workspace/results /workspace/datasets \
    /workspace/scripts /workspace/configs

WORKDIR /workspace

# Create a symlink so that benchmarks running fuse-overlayfs find /workspaces/Benchmarking
RUN mkdir -p /workspaces && ln -s /workspace /workspaces/Benchmarking

# Copy the init script and make it executable
COPY init.sh /init.sh
RUN chmod +x /init.sh

# Create non-root user for security (as recommended in the BenchExec guide)
RUN useradd -ms /bin/bash benchuser \
    && usermod -aG sudo benchuser \
    && echo "benchuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R benchuser:benchuser /workspace /opt/z3-versions

# # Install small utilities for QF_BV and download the QF_BV dataset
# RUN mkdir -p /benchmarks/QF_BV && wget -O /benchmarks/QF_BV/QF_BV.tar.zst "https://zenodo.org/records/11061097/files/QF_BV.tar.zst?download=1" \
#     && tar -I zstd -xf /benchmarks/QF_BV/QF_BV.tar.zst -C /benchmarks/QF_BV && rm /benchmarks/QF_BV/QF_BV.tar.zst

# # Install bzip2 and download the VLSAT-3 dataset in its own layer
# RUN mkdir -p /benchmarks/VLSAT3 \
#     && wget -r --no-parent -A "vlsat3*.smt2.bz2" https://cadp.inria.fr/ftp/benchmarks/vlsat/ -P /benchmarks/VLSAT3 \
#     && find /benchmarks/VLSAT3 -name "*.smt2.bz2" -exec bunzip2 {} +

# Set init.sh as the entrypoint (important to use array format)
ENTRYPOINT ["/init.sh"]

# Default command
CMD ["bash"]
