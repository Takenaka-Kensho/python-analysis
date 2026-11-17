from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import math
import f90nml
try: plt.style.use('./style-math.mplstyle')
except: pass

import read_gknet_nml

"""##################################################################################
    dispersion.py v6.1

    ***EXPLANATION***
    plot dispersion relation

    ***VERSION***
    v6.0 (2026/02) takenaka
    - change algorithm of gamma/omega calculation
    - compute window standard deviation
    v6.1 (2026/04) takenaka
    - modify elimination process
    - change text output format
    - improve usability
    v6.2 (2026/07) takenaka
    - new gknet.data format
    v6.3 (2026/11) takenaka
    - modify label k_y rho_ti -> k_theta rho_ti

    ***STRUCTURE***
    .
    ├──GKNET.data  (IN)
    ├──mode-z.txt  (IN)
    ├──omega-t.txt  (IN)
    ├──flux_length.txt  (IN)
    └──arbitary directory/
        ├──dispersion.py
        ├──dispersion_n.png  (OUT)
        ├──dispersion_m.png  (OUT)
        └──dispersion.txt  (OUT)
##################################################################################"""

@dataclass
class Config:
    version: str
    path_root: str
    time_ref: float
    window_len: float
    elim_thr: float
    plot_std: bool
    # Advanced settings ---------
    save_extension: str
    n_sigma: int
    is_omega_d4:bool
    is_y_from_zero:bool


def dispersion(config: Config) -> None:
    # Parameter from namelist --------------------------------------------------
    nml = read_gknet_nml.simple_gknet_nml(path=f'{config.path_root}/GKNET.data', version=config.version)
    dt      = nml.L_t/nml.N_mode_z_out

    # Calculate gamma/omega ----------------------------------------------------
    n, k_theta, gmm_ave, gmm_std, omg_ave, omg_std = gamma_omega(nml.N_wedge, nml.n_max, dt, config.window_len, config.time_ref, config.path_root)

    # Eliminate unconverged value ---------------------------------------------
    gmm_ave_el, gmm_std_el, omg_ave_el, omg_std_el = elim_value(gmm_ave, gmm_std, omg_ave, omg_std, config.elim_thr)

    # Plot dispersion ----------------------------------------------------------
    plot_dispersion(n, gmm_ave_el, gmm_std_el, omg_ave_el, omg_std_el,
                    mode=0, config=config)
    plot_dispersion(k_theta, gmm_ave_el, gmm_std_el, omg_ave_el, omg_std_el,
                    mode=1, config=config)

    # Save data ----------------------------------------------------------------
    header = f'n,{"m":>12},{"gamma":>12},{"omega":>12},{"gamma_std":>12},{"omega_std":>12}'
    out_data = np.vstack([n, k_theta, gmm_ave, omg_ave, gmm_std, omg_std])[:,1:]
    np.savetxt(f'./dispersion.txt', out_data.T, fmt=['%3d', '%12.5e', '%12.5e', '%12.5e', '%12.5e', '%12.5e'], header=header)
    return None


def elim_value(arr, arr_key, arr2, arr_key2, th:float):
    arr_cp = np.copy(arr)
    arr_key_cp = np.copy(arr_key)
    arr2_cp = np.copy(arr2)
    arr_key2_cp = np.copy(arr_key2)

    for i, key in enumerate(arr_key):
        if key > th:
            arr_cp[i]      = np.nan
            arr_key_cp[i]  = np.nan
            arr2_cp[i]     = np.nan
            arr_key2_cp[i] = np.nan
    for i, key in enumerate(arr_key2):
        if key > th:
            arr_cp[i]      = np.nan
            arr_key_cp[i]  = np.nan
            arr2_cp[i]     = np.nan
            arr_key2_cp[i] = np.nan

    return arr_cp, arr_key_cp, arr2_cp, arr_key2_cp


def plot_dispersion(x, gmm_ave, gmm_std, omg_ave, omg_std, mode:int, config: Config):
    plot_std        = config.plot_std
    n_sigma         = config.n_sigma
    is_omega_d4     = config.is_omega_d4
    ext             = config.save_extension
    is_y_from_zero  = config.is_y_from_zero

    if mode == 0:   # toroidal mode plot
        xlbl = r'$n$'
        sname='n'
    if mode == 1:   # poloidal mode plot
        xlbl = r'$k_\theta \rho_{ti0}$'
        sname='m'

    if is_omega_d4:
        omg_ave_plot = omg_ave / 4
        omg_std_plot = omg_std / 4
    else:
        omg_ave_plot = omg_ave
        omg_std_plot = omg_std

    fig = plt.figure(figsize=(8,8))
    ax  = fig.add_subplot()

    ax.plot(x[1:], gmm_ave[1:], color='red', marker='o', markeredgewidth=2.0,
            markerfacecolor='none', zorder=3)
    ax.plot(x[1:], omg_ave_plot[1:], color='blue', marker='^', markeredgewidth=2.0,
            markerfacecolor='none', ls='dashed', zorder=3)

    # Plot window standard deviation -------------------------------------------
    if plot_std:
        ax.fill_between(x[1:],
                        y1=gmm_ave[1:]+gmm_std[1:]*n_sigma,
                        y2=gmm_ave[1:]-gmm_std[1:]*n_sigma,
                        color='tomato', alpha=0.15, zorder=-2, lw=0.0)
        ax.fill_between(x[1:],
                        y1=(omg_ave_plot[1:]+omg_std_plot[1:]*n_sigma),
                        y2=(omg_ave_plot[1:]-omg_std_plot[1:]*n_sigma),
                        color='dodgerblue', alpha=0.15, zorder=-2, lw=0.0)

    # Set plot -----------------------------------------------------------------
    ax.set_xlabel(xlbl)
    if is_omega_d4:
        ax.set_ylabel(r'$\gamma$ (solid), $\omega/4$ (dashed) $[v_\text{ti0}/R_0]$')
    else:
        ax.set_ylabel(r'$\gamma$ (solid), $\omega$ (dashed) $[v_\text{ti0}/R_0]$')
    
    ax.set_xlim(0)
    if mode == 0: ax.set_xticks(x)

    if is_y_from_zero: ax.set_ylim(0)
    else: ax.axhline(y=0.0, color='k', zorder=-1, lw=1.0, ls='dotted')

    # Save plot ----------------------------------------------------------------
    fig.tight_layout(pad=0.1)
    fig.savefig(f'./dispersion_{sname}.{ext}', dpi=200)
    printc(f'saved "dispersion_{sname}.{ext}"', c='y')
    plt.close()
    return


def gamma_omega(N_wedge:int, n_max:int, dt:float, window_len=3.0, time_ref=-1.0,
                path_root = '..'):
    mode_z      = np.loadtxt(f'{path_root}/mode-z.txt')
    omega_t_raw = np.loadtxt(f'{path_root}/omega-t.txt')
    flux_length = np.loadtxt(f'{path_root}/flux_length.txt')
    window = int(window_len//dt)

    # Standard output ----------------------------------------------------------
    print('='*50)
    print(f'{"N_wedge":<17}:{N_wedge:6d}')
    print(f'{"n_max":<17}:{n_max:6d}')
    print(f'{"window length":<17}:{window_len:6.2f}')
    if time_ref == -1.0: end_time = mode_z[-1,0]
    else: end_time = time_ref
    print(f'{"end time":<17}:{end_time:6.1f}')
    print('='*50)

    if time_ref == -1.0:
        t    = mode_z[:,0]          # time for mode energy
        mode = mode_z[:,1:]         # mode energy
        omega_t = omega_t_raw[:,1:] # omega_t
    else:
        t_end_index = int(time_ref//dt)
        t    = mode_z[:t_end_index,0]          # time for mode energy
        mode = mode_z[:t_end_index,1:]         # mode energy
        omega_t = omega_t_raw[:t_end_index,1:] # omega_t

    n_arr   = np.asarray([(nidx)*N_wedge for nidx in range(n_max+1)])
    k_theta = np.asarray([((nidx)*N_wedge*2*math.pi*flux_length[nidx,2]/flux_length[nidx,1]) for nidx in range(n_max+1)])

    t_len = len(t)
    for nidx in range(n_max+1):
        # Calculate growth rate ------------------------------------------------
        nlog = np.log(mode[:,nidx])
        gmm  = np.zeros(shape=(t_len-4))
        for j in range(2, t_len-2):
            gmm[j-2] = (nlog[j-2] - 8*nlog[j-1] + 8*nlog[j+1] - nlog[j+2])/(12*dt)

        # Connect discontinuous points of omega_t ------------------------------
        for j in range(t_len-1):
            if omega_t[j+1,nidx] - omega_t[j,nidx] > 1.5:
                omega_t[j+1:,nidx] = omega_t[j+1:,nidx] - np.pi
            elif omega_t[j+1,nidx] - omega_t[j,nidx] < -1.5:
                omega_t[j+1:,nidx] = omega_t[j+1:,nidx] + np.pi
        
        # Calculate real frequency ---------------------------------------------
        omg  = np.zeros(shape=(t_len-4))
        for j in range(2, t_len-2):
            omg[j-2] = (omega_t[j-2,nidx] - 8*omega_t[j-1,nidx] + 8*omega_t[j+1,nidx] - omega_t[j+2,nidx])/(12*dt)

        if nidx==0:
            # Initialize gamma/omega array
            gmm_win = gmm[-window:]
            omg_win = omg[-window:]
        else:
            gmm_win = np.vstack([gmm_win, gmm[-window:]])
            omg_win = np.vstack([omg_win, omg[-window:]])

    # Calculate average and standard deviation ---------------------------------
    gmm_ave = np.average(gmm_win, axis=1)
    gmm_std = np.std(gmm_win, axis=1, ddof=1)
    omg_ave = np.average(omg_win, axis=1)
    omg_std = np.std(omg_win, axis=1, ddof=1)
    
    return n_arr, k_theta, gmm_ave, gmm_std, omg_ave, omg_std


def printc(st:str, c='w') -> None:
    if c == 'y': print(f'\033[33m{st}\033[0m')
    elif c == 'r': print(f'\033[31m{st}\033[0m')
    elif c == 'm': print(f'\033[35m{st}\033[0m')
    elif c == 'b': print(f'\033[36m{st}\033[0m')
    else: print(st)


def main() -> None:
    #***************************************************************************
    #**    USER SETTINGS                                                      **
    #***************************************************************************
    config = Config(
        version    = 'AB',  # GKNETの実行バージョン AB|PS
        path_root  = '..',  # 実行ディレクトリ(run.exeなど)へのパス
        time_ref   = -1,    # 計算の参照時間点 [R_0/v_{ti}] 
                            # -1とすると自動で最後の時刻に
        window_len = 3.0,   # 分散・誤差計算のための時間窓幅 [R_0/v_{ti}]
        elim_thr   = 0.8,   # 除去する点の誤差閾値 [v_{ti}/R_0]
        plot_std   = True,  # 誤差幅のプロット
        
        # Advanced settings ----------------------------------------------------
        save_extension = 'png', # png/jpg/svg/pdf...
        n_sigma    = 1,         # error bar [-(n_sigma*std), +(n_sigma*std)]
        is_omega_d4= True,      # plot omega/4 instead of omega
                                # this setting do not affect text output
        is_y_from_zero = False, # set ylim (0)
    )

    dispersion(config)


if __name__ == "__main__":
    main()
