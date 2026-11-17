from matplotlib import pyplot as plt
import numpy as np
import pandas
try: plt.style.use('./style-math.mplstyle')
except: None

import read_gknet_nml

"""##################################################################################
    param_read.py

    ***EXPLANATION***
    read /eq_data/parameter.txt and plot vs x

    ***VERSION***
    v3.0 (2026/07)
    - update parameter.txt format

    ***STRUCTURE***
    .
    ├──eq_data/
    |   └──parameter.txt (IN)
    └──arbitary directory/
        ├──param_read.py
        ├──q_in_gknet.png (OUT)
        └──profile_in_gknet.png (OUT)
##################################################################################"""

def main():
    
    nml = read_gknet_nml.simple_gknet_nml(path=f'../GKNET.data', version='AB')
    N_x    = nml.N_x
    N_spec = nml.N_spec

    path = '../eq_data/parameter.txt'
    spec = ['Electron', 'Ion', 'Ion2', 'Ion3']
    clrs = ['b', 'r', 'darkturquoise', 'gold']
    lss  = ['solid', 'dashed', 'dotted', 'dotted']
    
    fig = plt.figure(figsize=(8,10))
    ax_q  = fig.add_subplot(311)
    ax_n  = fig.add_subplot(312)
    ax_t  = fig.add_subplot(313)

    #x psi q
    data_q = np.loadtxt(path, dtype=np.float32, max_rows=N_x)
    x_q = data_q[:,0]
    psi = data_q[:,1]
    q   = data_q[:,2]
    q_max = np.max(q)

    ax_q.plot(x_q, q, c='olive', lw=2.5, zorder=2)

    y_max = 0.0
    for i in range(N_spec):
        data_i = np.loadtxt(path, dtype=np.float32, skiprows=(N_x+1)*(i+1)+1, max_rows=N_x)
        x  = data_i[:,0]
        T0 = data_i[:,1]
        n0 = data_i[:,2]

        ax_n.plot(x, n0, c=clrs[i], label=spec[i], ls=lss[i], lw=2.5, zorder=2)
        ax_t.plot(x, T0, c=clrs[i], label=spec[i], ls=lss[i], lw=2.5, zorder=2)

        y_max_tmp = max(np.max(n0), np.max(T0))
        if y_max_tmp > y_max: y_max = y_max_tmp

    # --- fill calculation box & plot y=1 --------------------------------------
    for ax in [ax_q, ax_n, ax_t]:
        ax.fill_between(x=[0.0, np.min(x_q)], y1=[0.0, 0.0], y2=[1E5, 1E5],
                        color='lightgray')
        ax.fill_between(x=[np.max(x_q), 1.0], y1=[0.0, 0.0], y2=[1E5, 1E5],
                        color='lightgray')
    ax_n.axhline(y=1.0, c='k', lw=1.0, ls='dotted', zorder=1)
    ax_t.axhline(y=1.0, c='k', lw=1.0, ls='dotted', zorder=1)

    # --- figure settings ------------------------------------------------------
    for ax in [ax_q, ax_n, ax_t]:
        ax.set_xticks(np.linspace(0.0, 1.0, 11))
        ax.tick_params(axis='both', pad=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r'Radial label $x$', fontsize=20)
    ax_q.set_ylim(0.0, q_max*1.1)
    ax_n.set_ylim(0.0, y_max_tmp*1.1)
    ax_t.set_ylim(0.0, y_max_tmp*1.1)

    ax_q.set_ylabel('Safety factor'         , fontsize=20)
    ax_n.set_ylabel('Normalized density'    , fontsize=20)
    ax_t.set_ylabel('Normalized temperature', fontsize=20)

    fig.tight_layout(pad=0.2)
    fig.savefig('./profile_gknet.png')

    plt.close()


if __name__ == "__main__":
    main()