using AMDGPU, ImplicitGlobalGrid, Plots

@views d_xa(A) = A[2:end  , :     ] .- A[1:end-1, :     ]
@views d_xi(A) = A[2:end  ,2:end-1] .- A[1:end-1,2:end-1]
@views d_ya(A) = A[ :     ,2:end  ] .- A[ :     ,1:end-1]
@views d_yi(A) = A[2:end-1,2:end  ] .- A[2:end-1,1:end-1]
@views  inn(A) = A[2:end-1,2:end-1]

@views function diffusion2D()
    # Physics
    lx, ly  = 10.0, 10.0                                # Length of domain in dimensions x, y and z
    lam     = 1.0                                       # Thermal conductivity
    Cp0     = 1.0
    # Numerics
    nx, ny  = 127, 127                                    # Number of gridpoints dimensions x, y and z
    nt      = 1e3                                       # Number of time steps
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, 1) # Initialize the implicit global grid
    println("Process $me selecting device $(AMDGPU.device())")
    dx, dy  = lx/nx_g(), ly/ny_g()                      # Space step in dimension x
    dt      = min(dx*dx,dy*dy)*Cp0/lam/4.1              # Time step for the 3D Heat diffusion
    # Array initializations
    dTdt    = AMDGPU.zeros(Float64, nx-2, ny-2)
    qx      = AMDGPU.zeros(Float64, nx-1, ny-2)
    qy      = AMDGPU.zeros(Float64, nx-2, ny-1)
    Cp      = Cp0.*AMDGPU.ones(Float64, nx, ny)
    T       = zeros(Float64, nx, ny)
    # Initial conditions
    T       = ROCArray([exp(-(x_g(ix,dx,T)+dx/2 -lx/2)^2 -(y_g(iy,dy,T)+dy/2 -ly/2)^2) for ix=1:size(T,1), iy=1:size(T,2)])
    # visu
    gr(); ENV["GKSwstype"]="nul"; !ispath("../output") && mkdir("../output")
    nx_v = (nx-2)*dims[1]
    ny_v = (ny-2)*dims[2]
    T_v  = zeros(nx_v, ny_v)
    T_nh = zeros(nx-2, ny-2)
    # Time loop
    for it = 1:nt
        qx    .= -lam .* d_xi(T)./dx                                           # Fourier's law of heat conduction: q_x   = -λ ∂T/∂x
        qy    .= -lam .* d_yi(T)./dy                                           # ...                               q_y   = -λ ∂T/∂y
        dTdt  .= 1.0./inn(Cp) .* (-d_xa(qx)./dx .- d_ya(qy)./dy)               # Conservation of energy:           ∂T/∂t = 1/cₚ (-∂q_x/∂x - ∂q_y/∂y)
        T[2:end-1,2:end-1] .= inn(T) .+ dt .* dTdt                             # Update of temperature             T_new = T_old + ∂T/∂t
        update_halo!(T)                                                        # Update the halo of T
    end
    T_nh .= Array(T[2:end-1,2:end-1])
    gather!(T_nh, T_v)
    if (me==0) heatmap(transpose(T_v)); png("../output/Temp_$(nprocs)_$(nx_g())_$(ny_g()).png"); end
    finalize_global_grid()                                                     # Finalize the implicit global grid
end

diffusion2D()
