import zarr
import numpy as np
from numcodecs import Blosc
from mpi4py import MPI

import read_gknet_nml

"""##################################################################################
    compress_data.py

    ***EXPLANATION***
    Data compression to zarr format
    Recommend to use batch job

    ***VERSION***
    v0.3 (2026/01)
    - prototype
    v1.0 (2026/09)
    - modify data structure
    - MPI
##################################################################################"""

### Initialize zarr ###
def init_zarr(path, shape, chunks, dtype=np.complex64, clevel=3):
    compressor = Blosc(cname='zstd', clevel=clevel, shuffle=Blosc.BITSHUFFLE)
    z = zarr.open(
        path,
        mode='w',
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        zarr_format=2
    )
    return z

def write_zarr(
        data_zarr_path:str,
        descrpt:tuple,
        N_t:int,
        n_max:int,
        N_x:int,
        N_tht:int,
    ):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"save {descrpt[0]}")

    # calculate MPI chunks
    n_width = int(np.ceil((n_max+1)/4))
    n_range = range(n_width*rank, min(n_width*(rank+1), n_max+1))
    n_stt = n_range[0]
    n_end = n_range[-1]+1
    #

    ## initialize zarr
    if rank == 0:
        zarr.create_array(
            data_zarr_path,
            shape=(N_t, n_max, N_x, N_tht),
            chunks=(10, n_width, N_x, N_tht),
            dtype=np.complex64,
            overwrite=True
        )

    comm.Barrier()

    z = zarr.open_array(data_zarr_path, mode='r+')

    ## metadata
    if rank == 0:
        z.attrs["dims"]        = ["time", "n_idx", "x", "theta"]
        z.attrs["description"] = descrpt[0]
        z.attrs["dtype"]       = "complex64"

    ## load data & save zarr
    for t in range(0, N_t):
        ## initialize 3d array
        data_3d = np.zeros(shape=(len(n_range), N_x, N_tht), dtype=np.complex64)
        ## load data
        # for n_idx in range(n_max):
        for idx, n_idx in enumerate(n_range):
            path     = f'../{descrpt[0]}/{n_idx:02d}{descrpt[1]}{t:04d}.txt'
            ptl_data = np.loadtxt(path, dtype=np.float32)
            ptl_re   = ptl_data[:,0]
            ptl_im   = ptl_data[:,1]
            ptl_cx   = ptl_re + 1.0j*ptl_im
            ptl_2d = np.reshape(ptl_cx, (N_tht, N_x)).T    # (x, tht)
            data_3d[idx,:,:] = ptl_2d                    # (n, x, tht)
        ## write zarr
        z[t,n_stt:n_end,:,:] = data_3d

    comm.Barrier()
    return None


if __name__ == '__main__':
    # --- SETTINGS -------------------------------------------------------------
    mode = 2     # 0:phi | 1:apl | 2:both
    # --- Read namelist --------------------------------------------------------
    nml        = read_gknet_nml.simple_gknet_nml(path=f'../GKNET.data')
    N_tht      = nml.N_z * nml.z_ratio
    dt_p_n_out = nml.L_t / nml.N_p_n_out

    # --- Search last time ------------------------------------------------------
    last_time = np.loadtxt('../mode-z.txt', dtype=np.float32)[-1,0]
    N_t       = int(last_time / dt_p_n_out)

    # --- Save data ------------------------------------------------------------
    if mode == 0 or mode == 2:   # phi
        write_zarr(
            data_zarr_path='phi.zarr',
            descrpt=("phi","P"),
            N_t=N_t,
            n_max=nml.n_max,
            N_x=nml.N_x,
            N_tht=N_tht,
            )

    if mode == 1 or mode == 2: # apl
        write_zarr(
            data_zarr_path='apl.zarr',
            descrpt=("apl","A"),
            N_t=N_t,
            n_max=nml.n_max,
            N_x=nml.N_x,
            N_tht=N_tht,
            )
