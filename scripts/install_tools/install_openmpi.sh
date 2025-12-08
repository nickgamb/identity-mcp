#!/bin/bash
# Install OpenMPI for DeepSpeed multi-GPU support
# This script needs to be run on the server with sudo access

echo "Installing OpenMPI for DeepSpeed..."
echo "This requires sudo access."

# Install OpenMPI
sudo apt-get update
sudo apt-get install -y openmpi-bin libopenmpi-dev

# Verify installation
if command -v mpirun &> /dev/null; then
    echo "OpenMPI installed successfully!"
    mpirun --version
else
    echo "WARNING: OpenMPI installation may have failed"
fi

