#!/bin/bash

source ./setenv.sh

julia --project -O3 --check-bounds=no diffusion_2D_ap.jl

# julia --project -O3 --check-bounds=no diffusion_2D_kp.jl

# julia --project -O3 --check-bounds=no diffusion_2D_perf.jl
