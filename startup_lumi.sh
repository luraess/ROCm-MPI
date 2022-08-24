#!/bin/bash

source ./scripts/setenv_lumi.sh

touch Project.toml

julia --project -e 'using Pkg; pkg"add https://github.com/JuliaParallel/MPI.jl#master";'

julia --project -e 'using Pkg; pkg"add MPIPreferences";'

julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary(; library_names=["libmpi_cray"], mpiexec="srun")'

julia --project -e 'using Pkg; pkg"add https://github.com/luraess/ImplicitGlobalGrid.jl#lr/amdgpu-0.4.x-support"; pkg"add AMDGPU";'

julia --project -e 'using Pkg; pkg"add Plots";'

julia --project -e 'using Pkg; Pkg.build()' # on LUMI, building AMDGPU needs to be done on a compute node but compute nodes don't have internet access
