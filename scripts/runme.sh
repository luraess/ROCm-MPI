#!/bin/bash

# select platform
source ./setenv_ault.sh
# source ./setenv_lumi.sh

# julia --project rocmaware_test_selectdevice.jl

julia --project -O3 --check-bounds=no diffusion_2D_ap.jl

# julia --project -O3 --check-bounds=no diffusion_2D_kp.jl

# julia --project -O3 --check-bounds=no diffusion_2D_perf.jl
