#!/bin/bash

# Crusher uses MI250X
# for more information: 
# https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#gpu-aware-mpi

module load craype-accel-amd-gfx90a
module load cray-mpich
module load rocm

# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1

# Use system provided rocm module capabilities
export JULIA_AMDGPU_DISABLE_ARTIFACTS=1
export IGG_ROCMAWARE_MPI=1

echo "ENV setup done"
