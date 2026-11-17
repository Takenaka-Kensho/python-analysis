from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import f90nml
import os
try: plt.style.use('./style-math.mplstyle')
except: pass

import read_gknet_nml

"""
■ GKNET側の出力について[GKNET260723 ~ 260928]
*バージョンにより大幅に変わるので注意*
(1) モーメント量の定義
  n,U,T,Q,Gはそれぞれ密度、流速、温度、エネルギーフラックス、粒子フラックスを表す
    n = velocity integral of (f0 + f1)
    U = velocity integral of (v_para * (f0 + f1)) / n
    T = velocity integral of (2/3 * mv^2/2 * (f0 + f1)) / n
    Q = velocity integral of (V * mv^2/2 * f1)
    G = velocity integral of (V          * f1)
  *ただしそれぞれ磁気面平均量
  エネルギー密度 W = vel. int. of (mv^2/2 * (f0 + f1))に対して、
  等方圧力ならば W = 3/2 p となり、p = nTが上の定義と合致していることがわかる

(2) 平衡/揺動、es/emおよびゾーナル成分の分離
  Q,Gはそれぞれ[N,Tes,Tem,0]の4種類を出力している
  - Nは0次の速度項に対応、それ以外が揺動に対する流束
  - Tesは静電ポテンシャル揺動を介して影響する揺動流束
  - Temはベクトルポテンシャル揺動を介して影響する揺動流束
  - 0は揺動のゾーナル成分による流束

(3) 規格化
  n_s, T_sを粒子種sに対する密度・温度の規格化量とする(典型的にはx=0.5での値)
  v_paraがv_tsで規格化されており、Vがv_tiで規格化されているため、
  n[n_s], U[v_ts], T[T_s], Q[n_s T_s v_ti], G[n_s v_ti]
  (rho*倍の処理をすることで)Q_GB単位にする場合もある

(4) GKNETバージョン問題
  - GKNET260723版より、Qの定義が物理量に近くなった
  - GKNET260723版から260928版現在まで、Q0およびG0の出力にバグが存在
    伴って、chi_turb出力およびD_turb出力にもバグが存在する
    (chi_turb, D_turbにはそれぞれより本質的なバグが存在するため、本コードでは使用しない)
  - 本コードのexclude_zonalをFalseにすると*Tes+*Temを乱流成分として計算する


■ 拡散係数の定義
(1) 慣習的な定義 
  熱伝導成分Q_condはエネルギーフラックスから粒子移動による熱対流を差し引いた量
    Q_cond = Q - (3/2)*T*G 
  *ただし、流体力学における熱伝導フラックスは、エンタルピー対流を除いた量のこともある:
    Q_cond_hy = Q - (5/2)*T*G

  (熱伝導による)熱拡散係数chiは以下で定義される
    Q_cond = -n*chi_cond*(dT/dr)
  実行熱拡散係数はエネルギーフラックスから直接計算した量を採用することもある
    Q = -n*chi_eff*(dT/dr)

  本コードでは基本的には実行拡散係数を計算するが、オプションを選択すれば伝導による熱拡散係数も計算できる

(2) 計量の扱い
  g^xx = g^rr (r:rho)により密度勾配および温度勾配に現れる計量 sqrt{g^xx} が求められる
  数値平衡を用いる場合には計量ファイルから、解析平衡の場合はネームリストからそれぞれ必要な情報を読み込んで計算する
  v2.0以上のread_gknet_nml.pyでなければ実行不可なので注意
"""


class NonlinearTransport():
    def __init__(self, nml, path, nx, ns, exclude_zonal=True, is_from_metric=False):
        self.nml  = nml
        self.path = path
        self.nx   = nx
        self.ns   = ns

        # --- Input data from T*.txt -------------------------------------------
        data = np.loadtxt(path, dtype=np.float32)

        self.x  = np.reshape(data[:,0],  (-1, nx))[0,:]
        self.t  = np.reshape(data[:,1],  (-1, nx))[:,0]
        self.n  = np.reshape(data[:,2],  (-1, nx))
        self.U  = np.reshape(data[:,3],  (-1, nx))
        self.T  = np.reshape(data[:,4],  (-1, nx))
        QN      = np.reshape(data[:,5],  (-1, nx))
        QT_es   = np.reshape(data[:,6],  (-1, nx))
        QT_em   = np.reshape(data[:,7],  (-1, nx))
        Q0      = np.reshape(data[:,8],  (-1, nx))
        GN      = np.reshape(data[:,9],  (-1, nx))
        GT_es   = np.reshape(data[:,10], (-1, nx))
        GT_em   = np.reshape(data[:,11], (-1, nx))
        G0      = np.reshape(data[:,12], (-1, nx))

        # chi_turb = np.reshape(data[:,13], (-1, nx))
        # D_turb   = np.reshape(data[:,14], (-1, nx))

        # --- Input data from GKNET.data ---------------------------------------
        gknet_nml  = f90nml.read('../GKNET.data')
        self.n_rat = gknet_nml['PARAM_CONSTANT']['nrs'][ns]/gknet_nml['PARAM_CONSTANT']['nrs'][1]     # n_s/n_i
        self.T_rat = gknet_nml['PARAM_CONSTANT']['Trs'][ns]/gknet_nml['PARAM_CONSTANT']['Trs'][1]     # T_s/T_i

        # --- Calculate base value ---------------------------------------------
        self.QT = (QT_es + QT_em)
        self.GT = (GT_es + GT_em)
        self.Q0 = Q0
        self.G0 = G0

        # self.chi_turb = chi_turb
        # self.D_turb   = D_turb

        if exclude_zonal:
            ## --- zonal component excluded ---
            self.Q_turb = self.QT - self.Q0
            self.G_turb = self.GT - self.G0
        else:
            ## --- zonal component included ---
            self.Q_turb = self.QT
            self.G_turb = self.GT

        self.Q_cond = self.Q_turb - (3/2)*self.T*self.G_turb

        # --- estimate initial saturation time ---
        self.sat_idx = self.estimate_init_saturation()
        
        twin = 0.5
        self.twin_idx = max(int(twin//self.nml.energy_out_time), 1)
        
        print('-'*50)
        print(f'N_s : {self.ns}')
        print(f'estimated saturation time [R_0/v_ti] : {self.t[self.sat_idx]:.2f}')

        self.calc_g_xx()

        self.is_from_metric = is_from_metric
        if is_from_metric:
            # --- Read metric --------------------------------------------------
            data_met  = np.loadtxt('./metric.txt', dtype=np.float32)
            g_xx_up_sqrt = data_met[:]
            self.g_sq = g_xx_up_sqrt
        return


    def estimate_init_saturation(self, twin=5) -> int:
        twin_idx = int(twin//self.nml.energy_out_time)
        # twin [R0/v_ti] : time window to estimate initial saturation

        Q_ave = np.average(self.Q_turb, axis=1) # x-average
        Q_th  = np.max(Q_ave)*0.8

        for i in range(len(self.t)):
            if Q_ave[i] > Q_th:
                start_idx = i
                break

        max_idx = np.argmax(Q_ave[start_idx:start_idx+twin_idx])
        return start_idx+max_idx


    def calc_g_xx(self):
        nml = self.nml

        if nml.num_eq_flag == 1:
            # numerical equilibrium
            nx       = nml.N_x
            nx_left  = nml.N_x_left
            nx_right = nml.N_x_right
            
            mtr  = read_gknet_nml.read_metric()

            g_xx_up = (mtr.g_rr_up) / (self.nml.R_0**2) # [rho_ti^2]

            J_tor = (abs(mtr.g_rr_dn*mtr.g_pp_dn*mtr.g_tt_dn
                    + mtr.g_rp_dn*mtr.g_pt_dn*mtr.g_tr_dn
                    + mtr.g_tr_dn*mtr.g_rp_dn*mtr.g_pt_dn
                    - mtr.g_rr_dn*mtr.g_pt_dn*mtr.g_pt_dn
                    - mtr.g_rp_dn*mtr.g_rp_dn*mtr.g_tt_dn
                    - mtr.g_tr_dn*mtr.g_pp_dn*mtr.g_tr_dn))**0.5
            
            J_tor = J_tor * (self.nml.R_0**3)
            S_2d = np.average(J_tor, axis=1)

            g_xx_up = np.average(g_xx_up*J_tor, axis=1)/S_2d
            self.g_xx_up = g_xx_up[nx_left+2:nx_left+nx+2]
        else:
            # analytical equilibrium
            nx       = nml.N_x
            a0_rhoti = nml.R_0 * nml.ratio_a
            self.g_xx_up = np.ones((nx), dtype=float)*(1/a0_rhoti**2)
        return


    def calc_flux(self):
        self.flux_x_average()
        self.calc_coeff()
        return
    

    def flux_x_average(self):
        # --- spacial average of Q_eff -----------------------------------------
        self.Q_eff_aavs, self.Q_eff_stds, mid_idx = self.avg_std(self.Q_turb)
        print(f'{"saturation point (Q_eff)":<28} : x = {self.x[mid_idx]:.2f}')

        # --- spacial average of G ---------------------------------------------
        self.G_aavs, self.G_stds, mid_idx = self.avg_std(self.G_turb)
        print(f'{"saturation point (G)":<28} : x = {self.x[mid_idx]:.2f}')

        # --- spacial average of Q_cond ----------------------------------------
        self.Q_cond_aavs, self.Q_cond_stds, mid_idx = self.avg_std(self.Q_cond)
        print(f'{"saturation point (Q_cond)":<28} : x = {self.x[mid_idx]:.2f}')
        return


    def calc_coeff(self):
        T_div = np.gradient(self.T, self.x, axis=1) * (self.g_xx_up[None,:])**0.5
        T_div = np.where(abs(T_div) > 1E-10, T_div, 1E-10)

        n_div = np.gradient(self.n, self.x, axis=1) * (self.g_xx_up[None,:])**0.5
        n_div = np.where(abs(n_div) > 1E-10, T_div, 1E-10)

        chi = - self.Q_turb / (self.n * T_div)
        D   = - self.G_turb / (n_div)

        chi_cond = - self.Q_cond / (self.n * T_div)

        self.chi = chi
        self.D   = D
        self.chi_cond = chi_cond

        # --- Calculate average and standard deviation -------------------------
        self.chi_eff_aavs , self.chi_eff_stds , _ = self.avg_std(chi)
        self.D_aavs       , self.D_stds       , _ = self.avg_std(D)
        self.chi_cond_aavs, self.chi_cond_stds, _ = self.avg_std(chi_cond)
        return None


    def plot_1d_tp(self, ax:plt.Axes, mode:int,
                   tname=None, clr='k', ls='solid', alpha=0.2, is_plot_std=True):
        if mode==0:
            # Energy flux
            f     = self.Q_eff_aavs * (self.n_rat * self.T_rat) # re-normalize to [n_i T_i v_ti]
            fstd  = self.Q_eff_stds * (self.n_rat * self.T_rat)
        if mode==1:
            # Particle flux
            f     = self.G_aavs * self.n_rat    # re-normalize to [n_i v_ti]
            fstd  = self.G_stds * self.n_rat
        if mode==2:
            # Energy diffusion coefficient
            f     = self.chi_eff_aavs
            fstd  = self.chi_eff_stds
        if mode==3:
            # Particle diffusion coefficient
            f     = self.D_aavs
            fstd  = self.D_stds
        if mode==4:
            # Heat flux
            f     = self.Q_cond_aavs * (self.n_rat * self.T_rat) # re-normalize to [n_i T_i v_ti]
            fstd  = self.Q_cond_stds * (self.n_rat * self.T_rat)
        if mode==5:
            # Heat diffusion coefficient
            f     = self.chi_cond_aavs
            fstd  = self.chi_cond_stds

        ax.plot(self.t, f, color=clr, lw=1.5, ls=ls, label=tname, zorder=1)
        if is_plot_std:
            ax.fill_between(self.t, y1=(f - fstd), y2=(f + fstd), color=clr, alpha=alpha, lw=0.0, zorder=-1)
        return


    def plot_2d_tp(self, fig_2d:plt.Figure, ax_2d:plt.Axes, mode:int, is_plot_satu_time=False, tname=None):
        if mode==0:
            Z  = self.Q_turb.T * (self.n_rat*self.T_rat)    # re-normalize to [n_i T_i v_ti]
            s  = 'Q'
            lbl = 'Energy flux'
            lmath = r'$Q_\text{eff} ~ [n_i T_i v_{ti}]$'
        if mode==1:
            Z  = self.G_turb.T * self.n_rat                 # re-normalize to [n_i v_ti]
            s  = 'G'
            lbl = 'Particle flux'
            lmath = r'$\Gamma ~ [n_i v_{ti}]$'
        if mode==2:
            Z     = self.chi.T
            lbl   = 'Energy diffusion coeff.'
            lmath = r'$\chi ~ [\rho_{ti} v_{ti}]$'
        if mode==3:
            Z     = self.D.T
            lbl   = 'Particle diffusion coeff.'
            lmath = r'$D ~ [\rho_{ti} v_{ti}]$'
        if mode==4:
            Z     = self.Q_cond.T * (self.n_rat*self.T_rat)
            lbl   = 'Heat flux'
            lmath = r'$Q_\text{cond} ~ [n_i T_i v_{ti}]$'
        if mode==5:
            Z     = self.chi_cond.T
            lbl   = 'Heat diffusion coeff.'
            lmath = r'$\chi_\text{cond} ~ [\rho_{ti} v_{ti}]$'

        vmin, vmax = np.min(Z), np.max(Z)
        vabs = max(abs(vmin), abs(vmax))

        cf = ax_2d.contourf(self.t, self.x, Z,
                            cmap='seismic', levels=201, vmin=-vabs, vmax=vabs)
        cb = fig_2d.colorbar(cf, ax=ax_2d, pad=0.01)

        # --- settings of colorbar ---
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        cb.formatter = formatter
        cb.update_ticks()

        cb.set_label(lbl+' '+lmath, fontsize=20, rotation=270, labelpad=30)
        cb.ax.tick_params(labelsize=14, length=3, direction='out')

        if is_plot_satu_time:
            ax_2d.axvline(x=self.t[self.sat_idx], color='lightgray', lw=1.5)
            ax_2d.text(x=self.t[self.sat_idx], y=self.x[-1]*0.9, s=r'$t_\text{sat} = $' + f'{self.t[self.sat_idx]:.1f}', fontsize=15)

        if tname != None:
            ax_2d.set_title(f'{tname}, {lbl}', fontsize=18, loc='left')

        ax_2d.set_xlabel(r'$t ~ v_{ti}/R_0$')
        ax_2d.set_ylabel(r'$x$')
        return


    def avg_std(self, arr:np.ndarray, step=3):
        """
        calculate average (avg) and standard derivation (std)
        standard derivation is about spacial window length (from 1 to maximum length)
        
        - output
        arr_aavs : average of average-flux
        arr_stds : standard derivation of average-flux (depend on window length)
        *both of output are 1d-array
        """
        mid_idx = np.argmax(np.average(arr[(self.sat_idx-self.twin_idx):(self.sat_idx+self.twin_idx), :],axis=0))

        x_len = len(arr[0,:])
        win_max = min(mid_idx, x_len-mid_idx)

        win_lens = range(1, win_max, step)

        avgs = []
        for win in win_lens:
            arr_ave = np.average(arr[:, mid_idx-win:mid_idx+win], axis=1)
            avgs.append(arr_ave)
        
        avgs = np.asarray(avgs, dtype=np.float32)
        arr_aavs = np.average(avgs, axis=0)
        arr_stds = np.std(avgs, axis=0, ddof=1)

        return arr_aavs, arr_stds, mid_idx
    

################################################################################

def plot_flux(nl_ele:NonlinearTransport, nl_ion:NonlinearTransport, root:str,
              is_plot_satu_time=False, is_plot_std:bool=True, is_plot_cond:bool=False):
    # --- 2d plot --------------------------------------------------------------
    fig = plt.figure(figsize=(12,18))
    axes:list[plt.Axes] = fig.subplots(4,1)

    nl_ele.plot_2d_tp(fig, axes[0], mode=0, tname='Electron',
                        is_plot_satu_time=is_plot_satu_time)
    nl_ion.plot_2d_tp(fig, axes[1], mode=0, tname='Ion'     ,
                        is_plot_satu_time=is_plot_satu_time)
    nl_ele.plot_2d_tp(fig, axes[2], mode=1, tname='Electron',
                        is_plot_satu_time=is_plot_satu_time)
    nl_ion.plot_2d_tp(fig, axes[3], mode=1, tname='Ion'     ,
                        is_plot_satu_time=is_plot_satu_time)

    fig.tight_layout(pad=0.2)
    fig.savefig(f'{root}/2d_flux.png', dpi=300)
    plt.close()

    # --- 1d plot --------------------------------------------------------------
    fig = plt.figure(figsize=(10,9))
    axes:list[plt.Axes] = fig.subplots(2,1)

    alpha = 0.2
    nl_ele.plot_1d_tp(axes[0], mode=0, tname='Electron', clr='r', ls='solid' , alpha=alpha, is_plot_std=is_plot_std)
    nl_ion.plot_1d_tp(axes[0], mode=0, tname='Ion'     , clr='b', ls='dotted', alpha=alpha, is_plot_std=is_plot_std)
    nl_ele.plot_1d_tp(axes[1], mode=1, tname='Electron', clr='r', ls='solid' , alpha=alpha, is_plot_std=is_plot_std)
    nl_ion.plot_1d_tp(axes[1], mode=1, tname='Ion'     , clr='b', ls='dotted', alpha=alpha, is_plot_std=is_plot_std)

    if is_plot_cond:
        nl_ele.plot_1d_tp(axes[0], mode=4, tname='Electron', clr='darkorange', ls='solid' , alpha=alpha, is_plot_std=is_plot_std)
        nl_ion.plot_1d_tp(axes[0], mode=4, tname='Ion'     , clr='darkturquoise', ls='dotted', alpha=alpha, is_plot_std=is_plot_std)

    if is_plot_satu_time:
        # plot saturation time based on electron data
        axes[0].axvline(x=nl_ele.t[nl_ele.sat_idx], lw=0.8, color='k', ls='dashed')
        axes[1].axvline(x=nl_ele.t[nl_ele.sat_idx], lw=0.8, color='k', ls='dashed')

    axes[0].set_title('Energy flux', fontsize=15)
    axes[0].set_ylabel(r'Energy flux  $Q ~ [n_{i}T_{i}v_{ti}]$')
    axes[1].set_title('Particle flux', fontsize=15)
    axes[1].set_ylabel(r'Particle flux  $\Gamma ~ [n_{i}v_{ti}]$')

    t_max = nl_ele.t[-1]
    for i in [0,1]:
        axes[i].tick_params(axis='x', pad=8)
        axes[i].tick_params(axis='y', pad=8)
        axes[i].set_xlim(0, t_max)
        axes[i].set_xlabel(r'$t ~ v_{ti}/R_0$')
        axes[i].legend(fontsize=15)
    
    axes[0].set_ylim(0)
    axes[1].axhline(y=0.0, c='k', lw=0.5, zorder=-5)
    
    fig.tight_layout(pad=0.2)
    fig.savefig(f'{root}/1d_flux.svg')
    plt.close()


def plot_diff_coeff(nl_ele:NonlinearTransport, nl_ion:NonlinearTransport, root:str,
                    is_plot_std:bool=True, is_plot_satu_time:bool=False, is_plot_cond:bool=False, is_text_out:bool=True):
    # --- 2d plot --------------------------------------------------------------
    fig = plt.figure(figsize=(12,18))
    axes:list[plt.Axes] = fig.subplots(4,1)

    nl_ele.plot_2d_tp(fig, axes[0], mode=2, is_plot_satu_time=True, tname='Electron')
    nl_ion.plot_2d_tp(fig, axes[1], mode=2, is_plot_satu_time=True, tname='Ion'     )
    nl_ele.plot_2d_tp(fig, axes[2], mode=3, is_plot_satu_time=True, tname='Electron')
    nl_ion.plot_2d_tp(fig, axes[3], mode=3, is_plot_satu_time=True, tname='Ion'     )

    fig.tight_layout(pad=0.2)
    fig.savefig(f'{root}/2d_coeff.png', dpi=300)
    plt.close()

    # --- 1d plot --------------------------------------------------------------
    fig = plt.figure(figsize=(10,9))
    axes:list[plt.Axes] = fig.subplots(2,1)
    alpha = 0.2
    nl_ele.plot_1d_tp(axes[0], mode=2, tname='Electron', clr='r', ls='solid' , alpha=alpha, is_plot_std=is_plot_std)
    nl_ion.plot_1d_tp(axes[0], mode=2, tname='Ion'     , clr='b', ls='dotted', alpha=alpha, is_plot_std=is_plot_std)
    nl_ele.plot_1d_tp(axes[1], mode=3, tname='Electron', clr='r', ls='solid' , alpha=alpha, is_plot_std=is_plot_std)
    nl_ion.plot_1d_tp(axes[1], mode=3, tname='Ion'     , clr='b', ls='dotted', alpha=alpha, is_plot_std=is_plot_std)

    if is_plot_cond:
        nl_ele.plot_1d_tp(axes[0], mode=5, tname='Electron', clr='darkorange', ls='solid' , alpha=alpha, is_plot_std=is_plot_std)
        nl_ion.plot_1d_tp(axes[0], mode=5, tname='Ion'     , clr='darkturquoise', ls='dotted', alpha=alpha, is_plot_std=is_plot_std)

    if is_plot_satu_time:
        # plot saturation time based on electron data
        axes[0].axvline(x=nl_ele.t[nl_ele.sat_idx], lw=0.8, color='k', ls='dashed')
        axes[1].axvline(x=nl_ele.t[nl_ele.sat_idx], lw=0.8, color='k', ls='dashed')

    axes[0].set_title('Energy diffusion coefficient', fontsize=15)
    axes[0].set_ylabel(r'Energy diffusion coeff.  $\chi ~ [\rho_{ti} v_{ti}]$')
    axes[1].set_title('Particle diffusion coefficient', fontsize=15)
    axes[1].set_ylabel(r'Particle diffusion coeff. $D ~ [\rho_{ti} v_{ti}]$')

    t_max = nl_ele.t[-1]
    for i in [0,1]:
        axes[i].tick_params(axis='x', pad=8)
        axes[i].tick_params(axis='y', pad=8)
        axes[i].set_xlim(0, t_max)
        axes[i].set_xlabel(r'$t ~ v_{ti}/R_0$')
        axes[i].legend(fontsize=15)
    
    axes[0].set_ylim(0)
    axes[1].axhline(y=0.0, c='k', lw=0.5, zorder=-5)

    fig.tight_layout(pad=0.2)
    fig.savefig(f'{root}/1d_coeff.svg')
    plt.close()

    # --- Text output ----------------------------------------------------------
    if is_text_out:
        data_out = np.vstack([nl_ele.t,
                              nl_ele.chi_eff_aavs, nl_ion.chi_eff_aavs,
                              nl_ele.chi_eff_stds, nl_ion.chi_eff_stds,
                              nl_ele.D_aavs  , nl_ion.D_aavs  ,
                              nl_ele.D_stds  , nl_ion.D_stds   ])
        header = f'{"time":<10}{"chi_eff_e":<13}{"chi_eff_i":<13}{"chi_eff_e std":<13}{"chi_eff_i std":<13}'\
                f'{"D_e":<13}{"D_i":<13}{"D_e std":<13}{"D_i std":<13}'
        np.savetxt(f'{root}/diff_coeff.txt', data_out.T,
                fmt=['%10.5f','%12.5e','%12.5e','%12.5e','%12.5e','%12.5e','%12.5e','%12.5e','%12.5e'],
                header=header)

    return


# def diffusion_coeff_from_output(nl_ele:NonlinearTransport, nl_ion:NonlinearTransport, root:str):
#     # NOT RECOMMENDED
#     # --- 2d plot --------------------------------------------------------------
#     fig = plt.figure(figsize=(12,18))
#     axes = fig.subplots(4,1)

#     nl_ele.plot_2d_tp(fig, axes[0], mode=2, tname='Electron')
#     nl_ion.plot_2d_tp(fig, axes[1], mode=2, tname='Ion')
#     nl_ele.plot_2d_tp(fig, axes[2], mode=3, tname='Electron')
#     nl_ion.plot_2d_tp(fig, axes[3], mode=3, tname='Ion')

#     fig.tight_layout(pad=0.2)
#     fig.savefig(f'{root}/2d_energy_coeff_out.png', dpi=300)
#     plt.close()

#     # --- 1d plot --------------------------------------------------------------
#     fig = plt.figure()
#     ax  = fig.add_subplot()

#     aavs, stds = avg_std(nl_ele.chi_turb, nl_ele.x_Q_idx)
#     ax.plot(nl_ele.t, aavs, color='b')

#     aavs, stds = avg_std(nl_ion.chi_turb, nl_ion.x_Q_idx)
#     ax.plot(nl_ion.t, aavs, color='r')

#     fig.savefig('1d_coeff_out.png')
#     plt.close()

#     return


def main():
    # --- Data input -----------------------------------------------------------
    nml = read_gknet_nml.simple_gknet_nml(path=f'../GKNET.data', version='AB')
    nx  = nml.N_x
    root = './nl'

    if not os.path.exists(root): os.mkdir(root)

    # --- Make instance --------------------------------------------------------
    nl_ele = NonlinearTransport(nml, path='../T0.txt', nx=nx, ns=0, exclude_zonal=False)
    nl_ele.calc_flux()

    nl_ion = NonlinearTransport(nml, path='../T1.txt', nx=nx, ns=1, exclude_zonal=False)
    nl_ion.calc_flux()

    # --- Plot flux ------------------------------------------------------------
    plot_flux(nl_ele, nl_ion, root, is_plot_satu_time=True, is_plot_std=True)

    # --- Plot diffusion coefficient -------------------------------------------
    plot_diff_coeff(nl_ele, nl_ion, root, is_plot_satu_time=True, is_plot_std=True)

    return None


if __name__ == '__main__':
    main()
