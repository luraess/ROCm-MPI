# ROCm-MPI
ROCm (-aware) MPI tests

### Proof of concept on CSCS's `ault` server

<img src="docs/poc_rocmaware.png" alt="rocm-aware mpi" width="600">

### 1000 diffusion steps on 4 MI50 GPUs

<img src="docs/Temp_4_252_252.png" alt="rocm-aware mpi" width="600">

### Current pkg version needed (to be updated upon PRs)

The following package versions are currently needed to run ROCm (-aware) MPI tests successfully:
- AMDGPU: https://github.com/JuliaGPU/AMDGPU.jl/pull/220
- MPI: https://github.com/JuliaParallel/MPI.jl/pull/547
- ImplicitGlobalGrid: https://github.com/eth-cscs/ImplicitGlobalGrid.jl/pull/35
