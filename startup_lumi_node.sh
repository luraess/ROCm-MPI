#!/bin/bash

source ./scripts/setenv_lumi.sh

julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary(; library_names=["libmpi_cray"], mpiexec="srun")'

julia --project -e 'using AMDGPU; AMDGPU.versioninfo()' # on LUMI, building AMDGPU needs to be done on a compute node but compute nodes don't have internet access
