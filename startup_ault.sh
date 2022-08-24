#!/bin/bash

source ./scripts/setenv_ault.sh

touch Project.toml

julia --project -e 'using Pkg; pkg"add https://github.com/JuliaParallel/MPI.jl#master";'

julia --project -e 'using Pkg; pkg"add MPIPreferences";'

julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'

julia --project -e 'using Pkg; pkg"add https://github.com/luraess/ImplicitGlobalGrid.jl#lr/amdgpu-0.4.x-support"; pkg"add AMDGPU";'

julia --project -e 'using Pkg; pkg"add Plots";'

julia --project -e 'using Pkg; Pkg.build()'
