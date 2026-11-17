from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import time
try: import zarr
except: pass

import read_gknet_nml
import mode_contour
version = 1.0
"""##################################################################################
    dmd.py

    ***EXPLANATION***
    DMD code

    ***VERSION***
    v0.3 (2026/02)
    - prototype
    - zarr input
    v1.0 (2026/09)
    - modify zarr format
    - read_gknket_nml.py v1.1
    - improve algorithm
##################################################################################"""

### User settings ###
def settings():
    n_index = 6             # toroidal mode number n = n_index*N_wedge
    t_stt_index = 100       # input start time (index of phi/apl file)
    t_end_index = 400       # input end time   (index of phi/apl file)
    rank = 10               # rank should be smaller than (t_end_index - t_stt_index)
    data_kind = 0           # 0:phi (electrostatic potential)
                            # 1:apl (parallel vector potential)
    is_zarr = True

    return n_index, t_stt_index, t_end_index, rank, data_kind, is_zarr


### Hermitian(adjoint) matrix ###
def hrt(matrix:np.ndarray) -> np.ndarray:
    return np.conjugate(matrix.T)


### Array composing ###
def arr_input(n:int, t_stt:int, t_end:int, data_kind:int) -> np.ndarray:
    is_data_exist = False

    print('loading data...')
    for t in tqdm(range(t_stt, t_end+1)):
        if data_kind==0: path = f'../phi/{n:02d}P{t:04d}.txt'
        if data_kind==1: path = f'../apl/{n:02d}A{t:04d}.txt'
        data_raw = np.loadtxt(path)

        if len(np.shape(data_raw)) != 2:
            cprint('Data shape error', c='red')
            cprint('-> please verify that imaginaly part of data is output correctly', c='red')
            raise TypeError()

        data_re = data_raw[:,0]         # real part of phi
        data_im = data_raw[:,1]         # imaginaly part of phi
        data_cmx = data_re + 1j*data_im # complex data
        
        if not is_data_exist:
            data = np.reshape(data_cmx, (-1, 1))
            is_data_exist = True
        else:
            data = np.hstack([data, np.reshape(data_cmx, (-1, 1))])

    print('data matrix size :', end=' ')
    cprint(f'{np.shape(data)[0]} (spatial) x {np.shape(data)[1]} (temporal)', c='yellow')
    return data


def zarr_input(n:int, t_stt:int, t_end:int, data_kind:int) -> np.ndarray:
    ### For ZARR!!!
    print('loading data...')
    if data_kind==0: data_raw=zarr.open('phi.zarr', mode='r')
    if data_kind==1: data_raw=zarr.open('apl.zarr', mode='r')

    data_pick = data_raw[t_stt:t_end+1, n, :, :]
    data = data_pick.transpose(0, 2, 1).reshape(data_pick.shape[0], -1).T

    print('data matrix size :', end=' ')
    cprint(f'{np.shape(data)[0]} (spatial) x {np.shape(data)[1]} (temporal)', c='yellow')
    return data


### DMD ###
def solve_dmd(data, n:int, t_stt:int, t_end:int, rank:int, data_kind:int):
    # init_x = data[:,0]
    X1 = data[:, :-1]
    X2 = data[:, 1:]

    print('SVD processing...')
    ## Find U,S,V, s.t. decompose the singlar values
    ## Note: S is 1d-vector of R+, (s1, s2,..., sn), s1 > s2 >...> sn
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)

    ## X1 is similary decomposed as X1 = Ur @ Sr @ hrt(Vr)
    Ur  = U[:,:rank]            # C
    Vhr = Vh[:rank,:]           # C
    Vr  = hrt(Vhr)
    Uhr = hrt(Ur)
    Sr  = S[:rank]  # 1d-array

    print('SVD is done!')

    ## Atilde = hrt(Ur) @ X2 @ Vr @ inv(Sr)
    Atilde  = (Uhr @ X2 @ Vr) / Sr
    eigs, W = np.linalg.eig(Atilde)     # eigen-values/functions

    ## Phi = X2 @ Vr @ inv(Sr) @ W
    Phi = ((X2 @ Vr) / Sr) @ W
    x0tilde = np.diag(Sr) @ Vhr[:,0]
    b   = np.linalg.solve(W @ np.diag(eigs), x0tilde)

    return Phi, eigs, b, S


### Sort eigenvalues/functions by gammas ###
def sort_by_gamma(Phi:np.ndarray, eigs:np.ndarray, dt:float):
    gmm = np.real(np.log(eigs))
    omg = np.imag(np.log(eigs))
    
    idx = np.argsort(gmm)[::-1]
    gmm_s = gmm[idx]
    omg_s = omg[idx]
    Phi_s = Phi[:,idx]

    print('='*50)
    print('DMD result :')
    print(f"{'index':>8}{'gamma':>15}{'omega':>15}{'omega/4':>15}")
    for i in range(len(idx)): 
        printstr = f'{i+1:>8}{gmm_s[i]/dt:15.6f}{omg_s[i]/dt:15.6f}{omg_s[i]/(4*dt):15.6f}'
        if gmm_s[i] > 0 : cprint(printstr, c='yellow')
        else : print(printstr)
    print('='*50)

    return Phi_s, gmm_s, omg_s


### Save results ###
def save_data(Phi:np.ndarray, gmm_s:np.ndarray, omg_s:np.ndarray, t_stt:int, t_end:int, n:int, N_wedge:int):
    print('saving results...')
    path_save = './dmd_mode'

    if not os.path.isdir(path_save):
        os.mkdir(path_save)

    nx, rank = np.shape(Phi)

    ### Save meta data and eigenvalues
    path_meta_out = f'{path_save}/dmd_settings_n{n:02d}.meta'
    with open(path_meta_out, mode='w') as mo:
        mo.write(f"#{'version':>14}{'header_len':>15}\n")
        mo.write(f"{version:15.1f}{8:15d}\n")
        mo.write(f"#{'n_idx':>14}{'n':>15}\n")
        mo.write(f"{n:15d}{n*N_wedge:15d}\n")
        mo.write(f"#{'start_idx':>14}{'end_idx':>15}\n")
        mo.write(f"{t_stt:15d}{t_end:15d}\n")
        mo.write(f"#{'rank':>14}{'nx':>15}\n")
        mo.write(f"{rank:15d}{nx:15d}\n")
        mo.write(f"#{'gamma':>14}{'omega':>15}\n")
        np.savetxt(mo, np.vstack([gmm_s, omg_s]).T, fmt='%15.6f')

    ### Save eigenfunctions
    for i in tqdm(range(rank)):
        phi_i = Phi[:,i]
        phi_i_re = np.real(phi_i)
        phi_i_im = np.imag(phi_i)
        np.savetxt(f'{path_save}/n{n:02d}mode{i+1:03d}.txt', np.vstack([phi_i_re, phi_i_im]).T, fmt='%15.5e')
    return None


### Plot eigenfunction ###
def plot_contour(nx, ntht, n, N_wedge, rank):
    print('saving eigenfunction contour...')
    mode_range = range(1,rank+1)

    xticks = [100,10]        # [major locator, minor locator]
    yticks = [100,10]
    shape2D = (ntht,nx)
    
    grid_file = '../RZ-table.txt'
    R,Z = mode_contour.read_grid(grid_file, shape2D)

    xlim =[np.min(R)-10,np.max(R)+10]
    ylim =[np.min(Z)-30,np.max(Z)+30]

    xlabel=r'$R - R_0\ [\rho_{\rm{i}}]$'
    ylabel=r"$Z\ [\rho_{\rm{i}}]$"

    for mode in tqdm(mode_range):
        title = f'n={n*N_wedge:02d},mode:{mode}'
        phi, _ = mode_contour.read_phi(f'./dmd_mode/n{n:02d}mode{mode:03d}.txt', shape2D)
        vmax = np.max(np.abs(phi))
        vlim =[-vmax, vmax]
        fname = f'./dmd_mode/n{n:02d}mode{mode:03d}.png'
    
        mode_contour.plotPolSlice(phi,R,Z,vlim,xlim,ylim,xticks,yticks,fname,
                        title=title,xlabel=xlabel,ylabel=ylabel,
                        plot_bndry=False, color_bg='k',
                        cmap='seismic')
    return None
        

### Cumulative contribution of singular values ###
def s_cont(S, rank):
    display_len = max(rank, 20)
    s_sum = np.sum(S)
    s_rate = []
    s_temp = 0
    for i in range(len(S)):
        s_temp += S[i]
        s_rate.append(s_temp/s_sum)

    fig,ax = plt.subplots()
    ax.plot(np.arange(1,display_len+1), s_rate[:display_len], marker='+')
    ax.axvline(x = rank, ls='dotted', color='k', zorder=-1)

    ax.set_xlim(0, display_len)
    ax.set_ylim(min(s_rate[:display_len]), 1.0)
    # ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'Singular values')
    ax.set_ylabel(r'Contribution')

    fig.tight_layout(pad=0.2)
    fig.savefig('./s_cont.png')
    plt.close()
    return None    


def cprint(st:str, c:str):
    ## if you don't want to use this,
    ## please replace 'cprint' with 'print' in the code
    if   c=='yellow': print(f'\033[33m{st}\033[0m')
    elif c=='red':    print(f'\033[31m{st}\033[0m')
    elif c=='blue':   print(f'\033[36m{st}\033[0m')
    elif c=='m':      print(f'\033[35m{st}\033[0m')
    else: print(st)
    return None


def main():
    start = time.perf_counter()
    n, t_stt, t_end, rank, data_kind, is_zarr = settings()

    ### Parameter from namelist of GKNET
    nml = read_gknet_nml.simple_gknet_nml(path=f'../GKNET.data', version='AB')
    N_wedge    = nml.N_wedge
    N_x        = nml.N_x
    N_z        = nml.N_z
    z_ratio    = nml.z_ratio
    L_t        = nml.L_t
    N_p_out    = nml.N_p_n_out
    dt = L_t / N_p_out

    ### Standard output
    print('='*50)
    print(f"{'n index':<17}:{n:>10}")
    print(f"{'n':<17}:{n*N_wedge:>10}")
    print(f"{'start time index':<17}:{t_stt:>10}")
    print(f"{'end time index':<17}:{t_end:>10}")
    print(f"{'rank':<17}:{rank:>10}")
    if data_kind==0: print(f"{'data':<17}:{'phi':>10}")
    if data_kind==1: print(f"{'data':<17}:{'apl':>10}")
    if is_zarr: print(f"{'data type':<17}:{'Zarr':>10}")
    else:       print(f"{'data type':<17}:{'text':>10}")
    print('')
    print(f"{'time step width':<17}:{dt:10.4f}")
    print('='*50)

    ### Data input
    if is_zarr: data = zarr_input(n, t_stt, t_end, data_kind)
    else:       data = arr_input(n, t_stt, t_end, data_kind)

    ### DMD calculation
    Phi, eigs, b, S = solve_dmd(data, n, t_stt, t_end, rank, data_kind)

    ### Cumulative contribution
    s_cont(S, rank)

    ### Sort eigenvalues & modes by gamma
    Phi_s, gmm_s, omg_s = sort_by_gamma(Phi, eigs, dt)

    ### Save data
    save_data(Phi_s, gmm_s, omg_s, t_stt, t_end, n, N_wedge)

    ### Contour plot
    plot_contour(N_x, N_z*z_ratio, n, N_wedge, rank)

    end = time.perf_counter()
    print(f'execution time = {end-start:8.2f} s')
    print('='*50)


if __name__ == '__main__':
    main()