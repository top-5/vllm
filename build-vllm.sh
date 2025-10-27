#!/usr/bin/env bash
set -euo pipefail

# Build vLLM from source using CUDA 12.8 and PyTorch cu128 wheels.
# The script validates the toolchain, upgrades build deps, and installs vLLM in editable mode.
#
# IMPORTANT: This script requires CUDA 12.8+ to compile Blackwell (SM 100) MoE kernels.
# Set CUDA_HOME and PATH before running if /usr/local/cuda-12.8 is not default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_CMD=("$PYTHON_BIN" -m pip)

# Ensure CUDA 12.8 toolkit is used
export PATH="/usr/local/cuda-12.8/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

log() {
  printf '[build-vllm] %s\n' "$*"
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1. Ensure CUDA 12.8 is installed at /usr/local/cuda-12.8"
}

require_cmd "$PYTHON_BIN"
require_cmd nvcc

NVCC_RELEASE=$(nvcc --version | awk '/release/{for(i=1;i<=NF;i++){if($i=="release"){print $(i+1)}}}' | sed 's/,//')
if [[ -z "$NVCC_RELEASE" ]]; then
  die "Unable to determine nvcc release version"
fi

if [[ "$NVCC_RELEASE" != 12.8* ]]; then
  die "Detected CUDA toolkit release $NVCC_RELEASE. Please install CUDA 12.8 before building."
fi

log "Using nvcc release $NVCC_RELEASE"

TORCH_CUDA_VERSION=$($PYTHON_BIN -c 'import torch, sys; print(getattr(torch.version, "cuda", ""))' 2>/dev/null || true)
if [[ -z "$TORCH_CUDA_VERSION" ]]; then
  die "PyTorch is not installed in the current environment. Install torch with CUDA 12.8 support (e.g. torch==2.9.0+cu128)."
fi

if [[ "$TORCH_CUDA_VERSION" != 12.8* ]]; then
  die "PyTorch reports CUDA $TORCH_CUDA_VERSION. Install a cu128 wheel before proceeding."
fi

log "PyTorch CUDA version $TORCH_CUDA_VERSION"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

log "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
log "MAX_JOBS=$MAX_JOBS"

cd "$REPO_ROOT"

log "Upgrading build dependencies"
"${PIP_CMD[@]}" install --upgrade pip setuptools wheel packaging ninja cmake >/dev/null

log "Installing vLLM in editable mode"
"${PIP_CMD[@]}" install -e . --no-build-isolation -v

log "Build completed successfully"
