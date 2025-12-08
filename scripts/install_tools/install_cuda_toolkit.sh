#!/bin/bash
# Install CUDA Toolkit 12.1 for DeepSpeed
# This script needs to be run on the server with sudo access

echo "Installing CUDA Toolkit 12.1..."
echo "This requires sudo access."

# Add NVIDIA package repositories
if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
fi
sudo dpkg -i cuda-keyring_1.1-1_all.deb || true
sudo apt-get update

# Install libtinfo5 from Ubuntu 22.04 (required by nsight-systems but not in 24.04)
echo "Installing libtinfo5 compatibility package..."
if ! dpkg -l | grep -q libtinfo5; then
    if [ ! -f libtinfo5_6.3-2ubuntu0.1_amd64.deb ]; then
        wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
    fi
    sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb || sudo apt-get install -f -y
fi

# Install CUDA toolkit essential components (nvcc and headers - avoids nsight-systems dependency)
echo "Installing CUDA toolkit essential components (nvcc, cudart-dev, libraries-dev)..."
sudo apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 cuda-libraries-dev-12-1 cuda-nvml-dev-12-1 || {
    echo "Some components failed, trying minimal install..."
    sudo apt-get install -y cuda-nvcc-12-1 cuda-cudart-dev-12-1 || {
        echo "ERROR: Failed to install essential CUDA components"
        exit 1
    }
}

# Set CUDA_HOME in ~/.bashrc if not already set
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Toolkit" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12.1" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "Added CUDA_HOME to ~/.bashrc"
fi

# Verify installation
if [ -f /usr/local/cuda-12.1/bin/nvcc ]; then
    echo "CUDA Toolkit installed successfully!"
    echo "CUDA_HOME: /usr/local/cuda-12.1"
    echo ""
    echo "To use it in this session, run:"
    echo "  export CUDA_HOME=/usr/local/cuda-12.1"
    echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
    echo ""
    echo "Or restart your shell to load from ~/.bashrc"
else
    echo "WARNING: CUDA toolkit installation may have failed"
    echo "Check /usr/local/cuda-12.1/bin/nvcc"
fi

