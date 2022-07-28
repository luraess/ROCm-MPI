# ROCm-MPI
ROCm (-aware) MPI tests on AMD GPUs

## Multi AMD-GPU results
#### Proof of concept on CSCS's `ault` server

<img src="docs/poc_rocmaware.png" alt="rocm-aware mpi" width="500">

#### 1000 diffusion steps on 4 MI50 GPUs

<img src="docs/Temp_4_252_252.png" alt="rocm-aware mpi" width="500">

## Getting started
Upon cloning the ROCm-MPI repo:
1. `cd ROCm-MPI`
2. `srun -n 1 --mpi=pmix ./startup.sh`
3. `cd scripts`
4. `srun -n 4 --mpi=pmix ./runme.sh`
5. check the image saved in `/output`

> Uncomment the execution lines in `runme.sh` to switch from array programming (ap) to kernel programming (kp) or performance-oriented (perf) examples.

:warning: Make sure to modify the [`scripts/setenv.sh`](scripts/setenv.sh) script accordingly to the MPI and ROCm "modules" available on the machine you plan to run on.

:bulb: You can switch to non ROCM-aware MPI by commenting out [`scripts/setenv.sh`](scripts/setenv.sh) L.11-17:

```bash
# ROCm-aware MPI
module load roc-ompi
export IGG_ROCMAWARE_MPI=1

# Standard MPI
# module load openmpi
# export IGG_ROCMAWARE_MPI=0
```

## Dependences (dev)
The following package versions are currently needed to run ROCm (-aware) MPI tests successfully (see also in [`startup.sh`](startup.sh)):
- AMDGPU.jl v0.3.5 and above: https://github.com/JuliaGPU/AMDGPU.jl
- MPI.jl `#master`: https://github.com/JuliaParallel/MPI.jl#master
- ImplicitGlobalGrid.jl dev: https://github.com/luraess/ImplicitGlobalGrid.jl#mpi-dev
