#!/bin/bash
# Persistent setup script for development environment
# This script runs every time the container starts and allows you to make
# persistent system modifications without rebuilding the Docker image

echo "Running persistent setup script..."

# Check if this is the first run
if [ ! -f /opt/persistent-setup/.setup_complete ]; then
    echo "First run detected. You can add your custom setup commands here."
    
    # Example: Install additional packages
    # sudo apt-get update
    # sudo apt-get install -y your-package-here
    
    # Example: Install additional Python packages
    # pip3 install --user your-python-package
    
    # Example: Configure environment
    # echo 'export MY_CUSTOM_VAR="value"' >> ~/.bashrc
    
    # Mark setup as complete
    touch /opt/persistent-setup/.setup_complete
    echo "Initial persistent setup complete."
else
    echo "Persistent setup already completed."
fi

# Add any commands that should run every time the container starts
echo "Running startup commands..."

# Example: Start a service, set environment variables, etc.
# export CUSTOM_PATH="/opt/my-tools:$PATH"

# Create empty dataset folders for all logics to avoid None patterns
mkdir -p datasets/QF_LIA datasets/QF_LRA datasets/QF_UFLIA datasets/QF_ABV datasets/QF_BV

echo "Persistent setup script finished."