using MPI
using AMDGPU
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
# select device
comm_l = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
rank_l = MPI.Comm_rank(comm_l)
device = AMDGPU.default_device_id!(rank_l+1)
gpu_id = AMDGPU.default_device_id()
# select device
size = MPI.Comm_size(comm)
dst  = mod(rank+1, size)
src  = mod(rank-1, size)
println("rank=$rank rank_loc=$rank_l (gpu_id=$gpu_id - $device), size=$size, dst=$dst, src=$src")
N = 4
send_mesg = ROCArray{Float64}(undef, N)
recv_mesg = ROCArray{Float64}(undef, N)
fill!(send_mesg, Float64(rank))
#rreq = MPI.Irecv!(recv_mesg, src,  src+32, comm)
MPI.Barrier(comm)
rank==0 && println("start sending...")
MPI.Sendrecv!(send_mesg, dst, 0, recv_mesg, src, 0, comm)
println("recv_mesg on proc $rank: $recv_mesg")
MPI.Barrier(comm)
rank==0 && println("done.")
