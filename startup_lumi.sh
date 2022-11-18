#!/bin/bash

source ./scripts/setenv_lumi.sh

julia --project -e 'using Pkg; Pkg.resolve()'

julia --project -e 'using Pkg; pkg"add https://github.com/luraess/ImplicitGlobalGrid.jl#lr/amdgpu-0.4.x-support";'

julia --project -e 'using Pkg; pkg"add AMDGPU#jps/dev";'

julia --project -e 'using Pkg; pkg"add GPUCompiler#vc/always_inline";'
