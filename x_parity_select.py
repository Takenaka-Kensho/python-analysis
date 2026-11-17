import numpy as np
import matplotlib.pyplot as plt
import pandas
import f90nml
import os
try: plt.style.use('./style-math.mplstyle')
except: pass

"""##################################################################################
    x_parity_select.py
    written by Kim and Takenaka

    ***EXPLANATION***
    use fourier transform in direction of poloidal angle
    and plot poloidal harmonics

    ***VERSION***
    v2.0 (2025/07)
    - selected poloidal mode numbers are sorted in legend
    - combine poloidal mode number with color
    - optimize the code to run on the PS

    ***STRUCTURE***
    .
    ├──GKNET.data
    ├──RZ-table.txt
    ├──phi/
    |   └──**P****.txt (IN)
    └──arbitary directory/
        ├──x_parity_select.py
        └──output plot
##################################################################################"""

if __name__ == "__main__":
    ######################################
    path_nml   = '../GKNET.data'
    path_rz    = '../RZ-table.txt'

    p_mode = 0      # 0: phi,  1: apl
    n_list = [6]    # (toroidal mode / wedge_number) list
    t_list = [50]   # time list
    i_r_th = 1      # 0:rをフーリエ変換, 1:thetaをフーリエ変換
    slct_num = 5             # 描画するプロット数

    c_palette = ['red', 'blue', 'darkturquoise', 'gold', 'k', 'm', 'lightgray', 'brown', 'lime']
    ######################################
    c_len = len(c_palette)

    if os.path.exists(path_nml):
        # read parameter from name list of GKNET
        input_nml = f90nml.read(path_nml)
        N_x       = input_nml['PARAM_GRID']['N_x']
        N_z       = input_nml['PARAM_GRID']['N_z']
        z_ratio   = input_nml['PARAM_GRID']['z_ratio']
        n_max     = input_nml['PARAM_GRID']['n_max']
        N_wedge   = input_nml['PARAM_GRID']['N_w']
    else: raise FileNotFoundError('GKNET.data does not exist')
    N_theta = N_z * z_ratio

    for j in n_list:
        for time in t_list:
            if(p_mode==0):txt = f'{j:02d}P{time:04d}'   # reference data
            if(p_mode==1):txt = f'{j:02d}A{time:04d}'   # reference data

            # input RZ-table
            rz = pandas.read_table(path_rz, header = None, sep = '\\s+')
            theta_raw = rz[0]
            radi_raw  = rz[1]

            r      = sorted(set(radi_raw))
            theta  = sorted(set(theta_raw))
            nx     = len(r)
            ntheta = len(theta)

            if(p_mode==0):p_file    = '../phi/' + txt + '.txt'
            if(p_mode==1):p_file    = '../apl/' + txt + '.txt'
            p = pandas.read_table(p_file, header=None, sep = '\\s+')
            # print(p)
            p = p[0] #pandasのため
            # print(p)
            ### data 
            data = np.zeros((nx, ntheta), dtype=np.complex128)
            iline=0
            for thetai in range(0,ntheta,1):
                for ri in range(0,nx,1):
                    data[ri,thetai]=p[iline]
                    iline=iline+1
            # フーリエ変換結果を格納する配列
            fourier_transform = np.zeros((nx, ntheta), dtype=complex)
            # フーリエ変換を実行
            if(i_r_th==1):
                for ri in range(0,nx):#thetaフーリエ変換
                    fourier_transform[ri,:] = np.fft.fft(data[ri,:])
            if(i_r_th==0):
                for thetai in range(0,ntheta):#rフーリエ変換
                    fourier_transform[:,thetai] = np.fft.fft(data[:,thetai])

            # 周波数成分を取得
            frequencies = np.fft.fftfreq(len(data), r[1] - r[0])

            # 寄与が大きい成分を抽出
            if(i_r_th==1):
                # ナイキスト数の関係でntheta//2までを評価
                ampl      = np.array([max(abs(np.real(fourier_transform[:,i]))) for i in range(ntheta//2)])
                amp_index = np.argsort(-ampl)
            elif(i_r_th==0):
                # ナイキスト数の関係でnx//2までを評価
                ampl      = np.array([max(abs(np.real(fourier_transform[i,:]))) for i in range(nx//2)])
                amp_index = np.argsort(-ampl)
            amp_index_sort = sorted(amp_index[:slct_num])

            for i in range(slct_num):
                c_num = amp_index_sort[i]%c_len
                if(i_r_th==1):plt.plot(r,    np.real(fourier_transform[:,amp_index_sort[i]])  ,label=f"$m$={amp_index_sort[i]}", c=c_palette[c_num])
                if(i_r_th==0):plt.plot(theta,np.real(fourier_transform.T[:,amp_index_sort[i]]),label=f"$m$={amp_index_sort[i]}", c=c_palette[c_num])

            ### all
            # if(i_r_th==1):plt.plot(r,np.real(fourier_transform),label=f"")
            # if(i_r_th==0):plt.plot(theta,np.real(fourier_transform.T),label=f"")
            plt.title(f'n={N_wedge*j}')
            # plt.xlim(0.4,0.6)
            plt.grid(True,axis="x",linestyle="dashed",lw=1)
            plt.grid(True,axis="y",linestyle="dashed",lw=1)
            if(i_r_th==1):plt.xlabel(r"$s(=r/a)$")
            if(i_r_th==0):plt.xlabel(r"$\theta*$")
            if(p_mode==0):plt.ylabel(r"${\rm Re} \phi ~ {\rm [T_i]}$")
            if(p_mode==1):plt.ylabel(r"${\rm Re} A_\parallel$")
            plt.legend()

            if(p_mode==0):plt.savefig(f'./n{N_wedge*j}_{j:02d}phi{time}.png',bbox_inches="tight",dpi=300)
            if(p_mode==1):plt.savefig(f'./n{N_wedge*j}_{j:02d}apl{time}.png',bbox_inches="tight",dpi=300)
            plt.clf()

            # ##################
            # with open('data.txt', 'w', encoding='utf-8') as file:
            #     file.write(f"{frequencies}\n")
            
