using AMDGPU, ImplicitGlobalGrid, Plots

macro d_xa(A,ix,iy)  esc(:( ($A[$ix+1,$iy+0] - $A[$ix,$iy+0]) )) end
macro d_ya(A,ix,iy)  esc(:( ($A[$ix+0,$iy+1] - $A[$ix+0,$iy]) )) end
macro d_xi(A,ix,iy)  esc(:( ($A[$ix+1,$iy+1] - $A[$ix,$iy+1]) )) end
macro d_yi(A,ix,iy)  esc(:( ($A[$ix+1,$iy+1] - $A[$ix+1,$iy]) )) end
macro inn(A,ix,iy)   esc(:( ($A[$ix+1,$iy+1]) )) end
macro all(A,ix,iy)   esc(:( ($A[$ix  ,$iy  ]) )) end

"""
# Fourier's law of heat conduction:
q_x    = -λ ∂T/∂x
qx    .= -lam .* d_xi(T)./dx    
qy    .= -lam .* d_yi(T)./dy 
"""
function Flux!(qx, qy, T, lam, _dx, _dy)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y  # only workgroupIdx starts at 1
    if (ix<=size(qx,1) && iy<=size(qx,2))
        @all(qx,ix,iy) = -lam * @d_xi(T,ix,iy) * _dx
    end
    if (ix<=size(qy,1) && iy<=size(qy,2))
        @all(qy,ix,iy) = -lam * @d_yi(T,ix,iy) * _dy
    end
    return
end

"""
# Conservation of energy:  
∂T/∂t  = 1/cₚ (-∂q_x/∂x - ∂q_y/∂y)
dTdt  .= 1.0./inn(Cp) .* (-d_xa(qx)./dx .- d_ya(qy)./dy)   
"""
function Residual!(dTdt, qx, qy, Cp, _dx, _dy)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y  # only workgroupIdx starts at 1
    if (ix<=size(dTdt,1) && iy<=size(dTdt,2))
        @all(dTdt,ix,iy) = 1.0/@inn(Cp,ix,iy) * - (@d_xa(qx,ix,iy) * _dx + @d_ya(qy,ix,iy) * _dy)
    end
    return
end

"""
# Update of temperature  
T[2:end-1,2:end-1] .= inn(T) .+ dt .* dTdt 
T_new               = T_old + ∂T/∂t    
"""
function Update!(T, dt, dTdt)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x  # only workgroupIdx starts at 1
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y  # only workgroupIdx starts at 1
    if (ix<=size(T,1)-2 && iy<=size(T,2)-2)
        @inn(T,ix,iy) += dt * @all(dTdt,ix,iy)
    end
    return
end

@views function diffusion2D()
    # Physics
    lx, ly  = 10.0, 10.0                                # Length of domain in dimensions x, y and z
    lam     = 1.0                                       # Thermal conductivity
    Cp0     = 1.0
    # Numerics
    blocs   = (32, 8, 1)
    grid    = (4, 16, 1)
    nx, ny  = blocs[1].*grid[1], blocs[2].*grid[2] # number of grid points
    gridsz  = (nx, ny, 1)
    nt      = 1e3                                       # Number of time steps
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, 1) # Initialize the implicit global grid
    println("Process $me selecting device $(AMDGPU.device())")
    dx, dy  = lx/nx_g(), ly/ny_g()                      # Space step in dimension x
    _dx,_dy = 1.0/dx, 1.0/dy
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
    me==0 && print("Start of time loop...")
    for it = 1:nt
        wait( @roc groupsize=blocs gridsize=gridsz Flux!(qx, qy, T, lam, _dx, _dy) )
        wait( @roc groupsize=blocs gridsize=gridsz Residual!(dTdt, qx, qy, Cp, _dx, _dy) )                            # ...                               q_y   = -λ ∂T/∂y
        wait( @roc groupsize=blocs gridsize=gridsz Update!(T, dt, dTdt) )
        update_halo!(T)                                                        # Update the halo of T
    end
    me==0 && println("End of time loop")
    T_nh .= Array(T[2:end-1,2:end-1])
    gather!(T_nh, T_v)
    if (me==0) heatmap(transpose(T_v)); png("../output/Temp_$(nprocs)_$(nx_g())_$(ny_g()).png"); end
    finalize_global_grid()                                                     # Finalize the implicit global grid
end

diffusion2D()
