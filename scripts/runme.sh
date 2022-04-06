#!/bin/bash

# module load hip-rocclr/4.2.0 hip/4.2.0 hsa-rocr-dev/4.2.0 hsakmt-roct/4.2.0 llvm-amdgpu/4.2.0 rocm-cmake/4.2.0 rocminfo/4.2.0 roctracer-dev-api/4.2.0
module load rocm hip-rocclr hip hsa-rocr-dev hsakmt-roct llvm-amdgpu rocm-cmake rocminfo roctracer-dev-api rocprofiler-dev rocm-smi-lib

export JULIA_AMDGPU_DISABLE_ARTIFACTS=0

module load roc-ompi
export SLURM_MPI_TYPE=pmix

# module load openmpi

export UCX_WARN_UNUSED_ENV_VARS=n
export JULIA_MPI_BINARY=system
export IGG_ROCMAWARE_MPI=1

export ROCR_VISIBLE_DEVICES=1,2

julia --project
# julia --project diffusion_2D.jl

# Status `/scratch/lraess/dev/test_IGG/Project.toml`
# [21141c5a] AMDGPU v0.3.4 `https://github.com/JuliaGPU/AMDGPU.jl.git#ee8f4b6`
# [4d7a3746] ImplicitGlobalGrid v0.13.0 `/scratch/lraess/dev/ImplicitGlobalGrid.jl/`
# [da04e1cc] MPI v0.19.2 `/scratch/lraess/dev/MPI.jl/`