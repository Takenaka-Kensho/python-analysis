from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import f90nml
try: plt.style.use('./style-math.mplstyle')
except: pass

import read_gknet_nml

"""##################################################################################
    energy_time.py

    ***EXPLANATION***
    generate time vs electrostatic energy plot

    ***VERSION***
    v6.0 (2026/04)
    - legend settings
    - improve usability
    v6.1 (2026/07)
    - new gknet.data format

    ***STRUCTURE***
    .
    ├──GKNET.data  (IN)
    ├──mode-z.txt  (IN)
    └──arbitary directory/
        ├──energy_time.py
        └──mode-evolution.png  (OUT)
##################################################################################"""

@dataclass
class Config:
    version: str
    path_root: str
    y_min: float # or -1
    save_extension: str

def energy_time(config: Config) -> None:
    # Read mode-z file ---------------------------------------------------------
    # e_t = pandas.read_table(f'{config.path_root}/mode-z.txt', header=None, sep = '\\s+')
    e_t = np.loadtxt(f'{config.path_root}/mode-z.txt', dtype=np.float32)
    
    # Read parameter from name list of GKNET -----------------------------------
    nml = read_gknet_nml.simple_gknet_nml(path=f'{config.path_root}/GKNET.data', version=config.version)
    n_max     = nml.n_max
    N_wedge   = nml.N_wedge
    nonlinear = nml.nonlinear
    
    print('='*50)
    print(f'{"n_max":<17}:{n_max:>8}')
    print(f'{"N_wedge":<17}:{N_wedge:>8}')
    print(f'{"nonlinear":<17}:{str(nonlinear):>8}')
    print('='*50)

    nt,nn=np.shape(e_t)
    if nn > n_max+2:
        printc('n_max value is invalid', c='r')
    
    # Initialize plot ----------------------------------------------------------
    fig = plt.figure(figsize=(12,8))
    ax  = fig.add_subplot(111)
    ax.set_xlabel(r'$t ~ v_{t0i}/R_0$')
    ax.set_ylabel(r'$|e \phi _{n}/T_{i}|$')
    ax.set_yscale('log')
    tmin=e_t[0,0]
    tmax=e_t[nt-1,0]
    ax.set_xlim(tmin,tmax)
    if config.y_min != -1:
        y_max = np.max(e_t[:,1:])
        ax.set_ylim(config.y_min, y_max)

    c_palette = ['red', 'blue', 'darkturquoise', 'gold', 'k', 'm', 'gray', 'brown', 'lime']
    s_palette = ['solid', 'dashed', 'dotted', 'dashdot', 'solid']
    c_len = len(c_palette)
    s_len = len(s_palette)
    if n_max+1 > c_len*s_len: 
        printc('c_palette or s_palette should be increased', c='r')

    # Plot each toroidal mode --------------------------------------------------
    if nonlinear: st = 1    # plot n >= 0
    else: st = 2            # plot only n > 0

    for i in range(st, n_max+2):
        label_txt = f'n = {N_wedge*(i-1)}'
        ax.plot(e_t[:,0], e_t[:,i], label = label_txt, 
                c=c_palette[(i-st)%c_len], ls=s_palette[(i-st)//c_len])
    
    # Save figure & finalize ---------------------------------------------------
    ax.legend(fontsize = 16,
              loc='upper left',
              bbox_to_anchor=(1.0, 1.01),
              frameon=False,
              ncol=max(int(n_max/10), 1))
    fig.tight_layout(pad=0.1)
    fig.savefig(f'./mode-evolution.{config.save_extension}',dpi=300)
    printc(f'saved "mode-evolution.{config.save_extension}"', c='y')
    plt.close()
    return None


def printc(st:str, c='w') -> None:
    if c == 'y': print(f'\033[33m{st}\033[0m')
    elif c == 'r': print(f'\033[31m{st}\033[0m')
    elif c == 'm': print(f'\033[35m{st}\033[0m')
    elif c == 'b': print(f'\033[36m{st}\033[0m')
    else: print(st)
    

def main():
    #***************************************************************************
    #**    USER SETTINGS                                                      **
    #***************************************************************************
    config = Config(
        version   = 'AB',       # GKNETの実行バージョン
        path_root = '..',       # 実行ディレクトリ(run.exeなど)へのパス
        y_min     = -1,         # -1 : auto
        save_extension = 'png', # png/jpg/svg/pdf...
    )

    energy_time(config)

    
if __name__ == "__main__":
    main()