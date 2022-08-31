using AMDGPU, ImplicitGlobalGrid, Plots, Printf

function diffusion_step!(T2, T, Cp, lam, dt, _dx, _dy, _dz)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    iz = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z
    nx, ny, nz = size(T2)
    if (ix > 1 && ix < nx && iy > 1 && iy < ny && iz > 1 && iz < nz)
        #! format:off
        @inbounds T2[ix,iy,iz] = T[ix,iy,iz] + dt*(Cp[ix,iy,iz]*(
                               - ((-lam*(T[ix+1,iy,iz] - T[ix,iy,iz])*_dx) - (-lam*(T[ix,iy,iz] - T[ix-1,iy,iz])*_dx))*_dx
                               - ((-lam*(T[ix,iy+1,iz] - T[ix,iy,iz])*_dy) - (-lam*(T[ix,iy,iz] - T[ix,iy-1,iz])*_dy))*_dy 
                               - ((-lam*(T[ix,iy,iz+1] - T[ix,iy,iz])*_dz) - (-lam*(T[ix,iy,iz] - T[ix,iy,iz-1])*_dz))*_dz ))
        #! format:on
    end
    return
end

function diffusion_step!(T2, T, Cp, lam, dt, _dx, _dy, _dz, b_width, istep)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    iz = (workgroupIdx().z - 1) * workgroupDim().z + workitemIdx().z
    nx, ny, nz = size(T2)
    # CommOverlap
    if (istep == 1 &&
        (ix > b_width[1] && ix < nx - b_width[1] && iy > b_width[2] &&
         iy < ny - b_width[2] && iz > b_width[3] && iz < nz - b_width[3]))
        @goto early_exit
    end
    if (istep == 2 &&
        (ix <= b_width[1] || ix >= nx - b_width[1] || iy <= b_width[2] ||
         iy >= ny - b_width[2] || iz <= b_width[3] || iz >= nz - b_width[3]))
        @goto early_exit
    end
    if (ix > 1 && ix < nx && iy > 1 && iy < ny && iz > 1 && iz < nz)
        #! format:off
        @inbounds T2[ix,iy,iz] = T[ix,iy,iz] + dt*(Cp[ix,iy,iz]*(
                               - ((-lam*(T[ix+1,iy,iz] - T[ix,iy,iz])*_dx) - (-lam*(T[ix,iy,iz] - T[ix-1,iy,iz])*_dx))*_dx
                               - ((-lam*(T[ix,iy+1,iz] - T[ix,iy,iz])*_dy) - (-lam*(T[ix,iy,iz] - T[ix,iy-1,iz])*_dy))*_dy 
                               - ((-lam*(T[ix,iy,iz+1] - T[ix,iy,iz])*_dz) - (-lam*(T[ix,iy,iz] - T[ix,iy,iz-1])*_dz))*_dz ))
        #! format:on
    end
    @label early_exit
    return
end

@views function diffusion3D(; do_vis = false)
    # Physics
    lx, ly, lz = 10.0, 10.0, 10.0                      # Length of domain in dimensions x, y and z
    lam = 1.0                                          # Thermal conductivity
    Cp0 = 1.0
    # Numerics
    fact = 6
    nx, ny, nz = fact * 128, fact * 128, fact * 128    # number of grid points
    threads = (32, 2, 2)
    grid = (nx, ny, nz)
    b_width = (32, 2, 2)
    nt = 1e2                                           # Number of time steps
    me, dims, nprocs = init_global_grid(nx, ny, nz)    # Initialize the implicit global grid
    println("Process $me selecting device $(AMDGPU.default_device_id())")
    dx, dy, dz = lx / nx_g(), ly / ny_g(), lz / nz_g() # Space step in dimension x
    _dx, _dy, _dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    dt = min(dx * dx, dy * dy, dz * dz) * Cp0 / lam / 6.1     # Time step for the 3D Heat diffusion
    # Array initializations
    Cp = Cp0 .* AMDGPU.ones(Float64, nx, ny, nz)
    T = zeros(Float64, nx, ny, nz)
    # Initial conditions
    # T       = ROCArray([exp(-(x_g(ix,dx,T)+dx/2 -lx/2)^2 -(y_g(iy,dy,T)+dy/2 -ly/2)^2 -(z_g(iz,dz,T)+dz/2 -lz/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)])
    T = ROCArray(rand(Float64, nx, ny, nz))
    T2 = copy(T)
    # visu
    if do_vis
        if (me == 0)
            gr()
            ENV["GKSwstype"] = "nul"
            !ispath("../output") && mkdir("../output")
        end
        nx_v, ny_v, nz_v = (nx - 2) * dims[1], (ny - 2) * dims[2],
                           (nz - 2) * dims[3]
        T_v = zeros(nx_v, ny_v, nz_v)
        T_nh = zeros(nx - 2, ny - 2, nz - 2)
    end
    qs = Vector{AMDGPU.ROCQueue}(undef, 2)
    for istep in 1:2
        qs[istep] = istep == 1 ?
                    ROCQueue(AMDGPU.default_device(); priority = :high) :
                    ROCQueue(AMDGPU.default_device())
    end
    signals = Vector{AMDGPU.ROCKernelSignal}(undef, 2)
    sig_real = [ROCSignal(), ROCSignal()]

    GC.enable(false) # uncomment for prof, mtp

    # Time loop
    me == 0 && print("Starting the time loop 🚀...")
    for it in 1:nt
        if (it == 11)
            tic()
        end
        for istep in 1:2
            AMDGPU.HSA.signal_store_screlease(sig_real[istep].signal[], 1)
        end

        # (1) original
        # signals[1] = @roc signal=sig_real[1] wait=false mark=false groupsize=threads gridsize=grid diffusion_step!(T2, T, Cp, lam, dt, _dx, _dy, _dz)
        # wait(signals[1])
        # # wait( @roc groupsize=threads gridsize=grid diffusion_step!(T2, T, Cp, lam, dt, _dx, _dy, _dz) )
        # update_halo!(T2)
        # T, T2 = T2, T

        # (2) new split kernel
        # for istep=1:2
        #     signals[istep] = @roc signal=sig_real[istep] wait=false mark=false groupsize=threads gridsize=grid queue=qs[istep] diffusion_step!(T2, T, Cp, lam, dt, _dx, _dy, _dz, b_width, istep)
        # end
        # for istep = 1:2
        #     wait(signals[istep])
        # end
        # update_halo!(T2)
        # T, T2 = T2, T

        # (3) comm/comp overlap
        for istep in 1:2
            signals[istep] = @roc signal=sig_real[istep] wait=false mark=false groupsize=threads gridsize=grid queue=qs[istep] diffusion_step!(T2,
                                                                                                                                               T,
                                                                                                                                               Cp,
                                                                                                                                               lam,
                                                                                                                                               dt,
                                                                                                                                               _dx,
                                                                                                                                               _dy,
                                                                                                                                               _dz,
                                                                                                                                               b_width,
                                                                                                                                               istep)
        end
        wait(signals[1])
        update_halo!(T2)
        wait(signals[2])
        T, T2 = T2, T
    end
    wtime = toc()
    me == 0 && println("done")

    GC.enable(true) # uncomment for prof, mtp

    A_eff = (2 + 1) / 2^30 * nx * ny * nz * sizeof(Float64)  # Effective main memory access per iteration [GB] (Lower bound of required memory access: Te has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    wtime_it = wtime / (nt - 10)                             # Execution time per iteration [s]
    T_eff = A_eff / wtime_it                                 # Effective memory throughput [GB/s]
    me == 0 &&
        @printf("Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s) \n",
                nt, wtime, round(T_eff, sigdigits = 3))
    if do_vis
        T_nh .= Array(T[2:(end - 1), 2:(end - 1), 2:(end - 1)])
        gather!(T_nh, T_v)
        # (me==0) && @show maximum(T_v)
        if (me == 0)
            heatmap(transpose(view(T_v, :, :, ceil(Int, nz_v / 2))))
            png("../output/Temp3D_hide_$(nprocs)_$(nx_g())_$(ny_g())_$(nz_g()).png")
        end
    end
    finalize_global_grid()  # Finalize the implicit global grid
end

diffusion3D(; do_vis = false)
