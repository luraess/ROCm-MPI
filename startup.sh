#!/bin/bash

source ./scripts/setenv.sh

touch Project.toml

julia --project -e 'using Pkg; pkg"add https://github.com/luraess/MPI.jl#lr/rocmaware-dev";'

julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'

julia --project -e 'using Pkg; pkg"add https://github.com/luraess/ImplicitGlobalGrid.jl#mpi-dev"; pkg"add AMDGPU";'

julia --project -e 'using Pkg; pkg"add Plots"'

julia --project -e 'using Pkg; Pkg.build()'
