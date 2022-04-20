#!/bin/bash

source ./setenv.sh

julia --project -O3 --check-bounds=no diffusion_2D.jl
