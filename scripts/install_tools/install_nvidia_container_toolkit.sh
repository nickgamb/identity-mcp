#!/bin/bash
# Install NVIDIA Container Toolkit for Docker GPU support
# Run with: sudo bash scripts/install_nvidia_container_toolkit.sh

set -e

echo "Installing NVIDIA Container Toolkit..."

# Clean up any existing broken repository file
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Add NVIDIA GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository - use the generic deb repository for Ubuntu/Debian
# This works for Ubuntu 24.04 and other recent versions
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/${ARCH} /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update package list
sudo apt-get update

# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use nvidia runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker daemon
sudo systemctl restart docker

echo "NVIDIA Container Toolkit installed successfully!"
echo "Verifying installation..."
docker info | grep -i runtime

