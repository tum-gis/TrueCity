#!/bin/bash
# Build script for CUDA operations on node21
# This script detects CUDA installation and builds all required CUDA extensions

set -e

# Detect CUDA installation
if [ -z "$CUDA_HOME" ]; then
    # Try to find nvcc
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        echo "Detected CUDA_HOME from nvcc: $CUDA_HOME"
    # Try common CUDA paths
    elif [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
        echo "Detected CUDA_HOME from /usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_HOME="/opt/cuda"
        echo "Detected CUDA_HOME from /opt/cuda"
    else
        echo "ERROR: CUDA not found. Please set CUDA_HOME manually or load a CUDA module."
        echo "Try: module load cuda/12.1  (or your CUDA version)"
        exit 1
    fi
fi

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Using CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')"

# Verify CUDA headers exist
if [ ! -f "$CUDA_HOME/include/cuda_runtime_api.h" ]; then
    echo "ERROR: cuda_runtime_api.h not found at $CUDA_HOME/include/cuda_runtime_api.h"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build pointops
echo ""
echo "=========================================="
echo "Building pointops..."
echo "=========================================="
cd libs/pointops
python setup.py install
cd ../..

# Build pointops2
echo ""
echo "=========================================="
echo "Building pointops2..."
echo "=========================================="
cd libs/pointops2
python setup.py install
cd ../..

# Build pointgroup_ops (optional, but good to have)
if [ -d "libs/pointgroup_ops" ]; then
    echo ""
    echo "=========================================="
    echo "Building pointgroup_ops..."
    echo "=========================================="
    cd libs/pointgroup_ops
    python setup.py install
    cd ../..
fi

# Build pointseg (optional, but good to have)
if [ -d "libs/pointseg" ]; then
    echo ""
    echo "=========================================="
    echo "Building pointseg..."
    echo "=========================================="
    cd libs/pointseg
    python setup.py install
    cd ../..
fi

echo ""
echo "=========================================="
echo "All CUDA extensions built successfully!"
echo "=========================================="

