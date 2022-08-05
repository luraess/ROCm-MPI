#!/bin/bash

# module load craype-accel-amd-gfx908 # MI100
module load craype-accel-amd-gfx90a # MI250x
module load cray-mpich
module load rocm

export JULIA_AMDGPU_DISABLE_ARTIFACTS=1

# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_ROCMAWARE_MPI=1

echo "ENV setup done"
