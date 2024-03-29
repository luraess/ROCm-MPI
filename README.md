# ROCm-MPI
ROCm (-aware) MPI tests on AMD GPUs on following platforms:
- [LUMI-G supercomputer (MI250x)](#csc-lumi-g)
- [Ault test system (MI50)](#cscs-ault)
- [Crusher - Frontier's test bed (MI250x)](#olcf-crusher)

## Multi AMD-GPU results (on LUMI-G eap)

### 1000 diffusion steps on 4 MI250x GPUs
<img src="docs/Temp_ap_4_254_254_lumi.png" alt="rocm and mpi" width="500">

### Communication/computation overlap for ideal multi-GPU weak scaling
<img src="docs/weak_scale_lumi.png" alt="weakscaling LUMI" width="500">

Ideal weak scaling is achieved by overlapping computation with MPI communication
<img src="docs/hide_comm_diff3D_8gpus.png" alt="rocm and mpi" width="500">

The red-square highlights the asynchronous thus overlapping behaviour of the MPI point-to-point communication kernels and the physics computations (bottom trace). Early results achieve 97% of parallel efficiency among 16 MI250x GPUs on 2 LUMI-G eap nodes.


## Getting started

On all machines, download and install Julia v1.9 (nightly) on scratch, make sure to set `export JULIA_DEPOT_PATH` to point to a location on scratch.

### CSC LUMI-G
1. First `salloc -n 4 --gpus=4 -p eap -A project_XX --time=01:00:00`. Then, upon cloning the ROCm-MPI repo:
2. `cd ROCm-MPI`
3. `srun -n 1 ./startup_lumi.sh` _(note that compute nodes have no internet connexion but AMDGPU and MPI need to be built on a compute node...)_
4. `cd scripts`
5. `srun -n 4 ./runme.sh` making sure to include the `setenv_lumi.sh` in there
6. check the image saved in `/output`

:bulb: You can switch to non ROCM-aware MPI by setting ENV vars to 0 in [`scripts/setenv_lumi.sh`](scripts/setenv_lumi.sh) L.11-12:

```
# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_ROCMAWARE_MPI=1
```

### CSCS Ault
1. First `salloc -n 4 -p amdvega -w ault20 --gres=gpu:4 -A cXX --time=04:00:00`. Then, upon cloning the ROCm-MPI repo:
2. `cd ROCm-MPI`
3. `srun -n 1 --mpi=pmix ./startup_ault.sh`
4. `cd scripts`
5. `srun -n 4 --mpi=pmix ./runme.sh` making sure to include the `setenv_ault.sh` in there
6. check the image saved in `/output`

:bulb: You can switch to non ROCM-aware MPI by switching comments in [`scripts/setenv_ault.sh`](scripts/setenv_ault.sh) L.12-19:

```bash
# ROCm-aware MPI
module load roc-ompi
export IGG_ROCMAWARE_MPI=1

# Standard MPI
# export PMIX_MCA_psec=native
# module load openmpi
# export IGG_ROCMAWARE_MPI=0
```

### OLCF Crusher
1. First allocate some GPU resources. Then, upon cloning the ROCm-MPI repo:
2. `cd ROCm-MPI`
3. `srun -n 1 ./startup_crusher.sh`
4. `cd scripts`
5. `srun -n 4 ./runme.sh` making sure to include the `setenv_crusher.sh` in there
6. check the image saved in `/output`

:bulb: You can switch to non ROCM-aware MPI by setting ENV vars to 0 in [`scripts/setenv_crusher.sh`](scripts/setenv_crusher.sh) L.12 & 16:

```
export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_ROCMAWARE_MPI=1
```


## Misc

> Uncomment the execution lines in `runme.sh` to switch from array programming (ap) to kernel programming (kp) or performance-oriented (perf) examples.

:warning: Make sure to modify the `scripts/setenv_[...].sh` script accordingly to the MPI and ROCm "modules" available on the machine you plan to run on.

### Profiling
The profiling timeline can be generated upon running one of the `diffusion_{2,3}D_perf_hidecomm_prof.jl` scripts as:

```
rocprof --hsa-trace julia --project -O3 --check-bounds=no diffusion_{2,3}D_perf_hidecomm_prof.jl
```

> Note that because of a `LD_PRELOAD` conflict, a custom version of `rocprof` needs to be currently used on LUMI-G eap.

## Dependences (dev)
The following package versions are currently needed to run ROCm (-aware) MPI tests successfully (see also in [`startup.sh`](startup.sh)):
- AMDGPU.jl v0.4.4 on `#jps/dev`: https://github.com/JuliaGPU/AMDGPU.jl#jps/dev
- GPUCompiler v0.16.6 on `#vc/always_inline`: https://github.com/JuliaGPU/GPUCompiler.jl#vc/always_inline
- MPI.jl from registry: https://github.com/JuliaParallel/MPI.jl
- ImplicitGlobalGrid.jl `#lr/amdgpu-0.4.x-support`: https://github.com/luraess/ImplicitGlobalGrid.jl#lr/amdgpu-0.4.x-support
