from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
from tqdm.auto import tqdm
import matplotlib as mpl
import os
try: import zarr
except: pass

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mpl_toolkits.axes_grid1  as mtag1
from matplotlib import gridspec, rcParams;
try: plt.style.use('./style-math.mplstyle')
except: pass

import read_gknet_nml

"""##################################################################################
    mode_contour.py

    ***VERSION***
    v4.0 (2026/02)
    - plot from zarr data (use python 3.12)
    v4.1 (2026/04)
    - improve usability
    - change functions
    v4.2 (2026/07)
    - new GKNET.data format
    v4.3 (2026/09)
    - fix small bug

    ***STRUCTURE***
    .
    ├──GKNET.data
    ├──RZ-table.txt
    ├──phi or apl/
    |   └──**P****.txt | **A****.txt (IN)
    └──arbitary directory/
        ├──mode_contour.py
        └──phi or apl/
            └──**P****.png | **A****.png (OUT)
##################################################################################"""
rcParams['legend.labelspacing'] = 0.01
rcParams['legend.columnspacing'] = 0.01
rcParams['legend.borderpad'] = 0.00
rcParams['lines.linewidth'] = 2
rcParams['figure.subplot.left'] = 0.18
rcParams['figure.subplot.right'] = 0.8
rcParams['figure.autolayout'] = False
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

@dataclass
class Config:
    version: str
    path_root: str
    pmode: int
    t_range: Sequence
    n_range: Sequence
    xticks: list
    yticks: list
    cmap: str
    color_bg: str
    figsize: tuple
    plot_bndry: bool
    plot_metric: bool
    plot_metric_theta: bool
    is_zarr: bool


def fmt(x, pos):
    """
    set label of colorbar  a.aa \times 10^{b}
    
    """
    a, b = '{0:+.2e}'.format(x).split('e')
    b = int(b)
    if b < -15:
        return r'$\pm 0.0 \times 10^{0}$'
    else:
        return r'${0} \times 10^{{{1}}}$'.format(a, b)

def read_grid(grid_file, shape2D):
    """
    1) read 1D array of grid data, eletrostatic potential
    2) return 2D arrays of them with shape of [N_x,N_theta]
       and filename
    """
    
    # read R, Z data as 1D array 
    grid = np.loadtxt(grid_file, dtype=np.float64)

    # reshape R(i), Z(i) to R(x,theta), Z(x,theta)
    R_2d = np.array(grid[:,2],dtype=np.float64).reshape(shape2D).T
    Z_2d = np.array(grid[:,3],dtype=np.float64).reshape(shape2D).T
    
    return R_2d, Z_2d


def read_phi(pfile, shape2D):    
    # read phi as 1D array 
    p_1d = np.loadtxt(pfile)
    if len(np.shape(p_1d)) != 1:
        p_1d = p_1d[:,0]

    # reshape phi(i), Z(i) to phi(x,theta)
    p_2d = np.array(p_1d,dtype=np.float64).reshape(shape2D).T

    # set name of png file
    p_outfile = pfile.replace('txt', 'png')[1:]
        
    return p_2d, p_outfile


def plotPolSlice(var,xcrd,ycrd,vlim,xlim,ylim,xticks,yticks,fname,
                 title=None,xlabel=None,ylabel=None,
                 plot_bndry=True,cmap='jet',color_bg='none',cnum=201,tnum=11,
                 ipath='./',tag='_',figsize=(12,8),
                 plot_metric=False, plot_metric_theta=False,
                 R_met=None, Z_met=None):
        
    # Fill data from N_theta-1 to N_theta --------------------------------------
    xcrd = np.concatenate([xcrd, xcrd[:,0:1]],axis=1)
    ycrd = np.concatenate([ycrd, ycrd[:,0:1]],axis=1)
    var  = np.concatenate([var,   var[:,0:1]],axis=1)

    # Set interval and ticks for colormap --------------------------------------
    interval = (vlim[1]-vlim[0])*np.arange(cnum)/(cnum-1.) + vlim[0]
    ticks    = (vlim[1]-vlim[0])*np.arange(tnum)/(tnum-1.) + vlim[0]

    # Initialize figure --------------------------------------------------------
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=0.1,wspace=0.0)
    gs  = gridspec.GridSpec(1,2,width_ratios=(1,0.1),height_ratios=(1,))
    ax  = [plt.subplot(gs[0,0]), plt.subplot(gs[0,1])]

    # Plot contour ------------------------------------------------------------- 
    ax[0].set_aspect('equal')
    if title  != None: ax[0].set_title(title,y=1.01,fontsize=20)
    if xlabel != None: ax[0].set_xlabel(xlabel)
    if ylabel != None: ax[0].set_ylabel(ylabel)

    ax[0].set_facecolor(color_bg)
    ax[0].set_xlim(xlim[0],xlim[1])
    ax[0].set_ylim(ylim[0],ylim[1])
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(xticks[0]))
    ax[0].xaxis.set_minor_locator(ticker.MultipleLocator(xticks[1]))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(yticks[0]))
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(yticks[1]))

    if plot_bndry:
        ax[0].plot(xcrd[-1,:], ycrd[-1,:],'k-') # outer boundary 
        ax[0].plot(xcrd[ 0,:], ycrd[ 0,:],'k-') # inner boundary 

    cs0 = ax[0].contourf(xcrd,ycrd,var,interval,cmap=cmap)

    # Plot magnetic contour ----------------------------------------------------
    if plot_metric:
        nss, ntht = np.shape(R_met)
        for i in np.linspace(0,1,11)[1:-1]:
            x_ref = int(nss*i)
            ax[0].plot(R_met[x_ref,:], Z_met[x_ref,:], ls='solid', lw=0.5, c='k')
        ax[0].plot(R_met[-1,:], Z_met[-1,:], ls='solid', lw=1.0, c='k')
        if plot_metric_theta:
            for i in np.linspace(0,1,9)[:-1]:
                z_ref = int(ntht*i)
                ax[0].plot(R_met[:,z_ref], Z_met[:,z_ref], c='k', ls='dashed', lw=0.5)
    
    # Erase frames ------------------------------------------------------------- 
    ax[1].tick_params(left=False,  labelleft=False,
                      bottom=False,labelbottom=False,
                      right=False, labelright=False,
                      top=False,   labeltop=False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)

    # Plot color bar -----------------------------------------------------------
    divider = mtag1.make_axes_locatable(ax[1])
    cax = divider.append_axes('left','30%', pad='10%')
    cb0 = plt.colorbar(ax = ax[1], mappable=cs0, ticks=ticks,cax= cax,
                       extend='both',format=ticker.FuncFormatter(fmt))
    
    plt.draw()
    plt.savefig(fname,dpi=200)
    plt.close()


def read_metric(nx, nz, path, R_0):
    data = np.loadtxt(path)
    R     = np.reshape(data[:,17],(nx,nz))
    Z     = np.reshape(data[:,18],(nx,nz))

    R_st, Z_st = R[:,0].reshape(-1,1), Z[:,0].reshape(-1,1)
    R = np.hstack([R, R_st])
    Z = np.hstack([Z, Z_st])
    return R*R_0 - R_0, Z*R_0


def mode_contour(config: Config):
    path_root = config.path_root
    pmode = config.pmode
    t_range = config.t_range
    n_range = config.n_range

    xticks = config.xticks
    yticks = config.yticks
    cmap = config.cmap
    figsize = config.figsize
    color_bg = config.color_bg
    plot_bndry = config.plot_bndry
    plot_metric = config.plot_metric
    plot_metric_theta = config.plot_metric_theta
    is_zarr = config.is_zarr

    # Read namelist ------------------------------------------------------------
    nml = read_gknet_nml.simple_gknet_nml(path=f'{path_root}/GKNET.data', version=config.version)
    metric_path = '../' + nml.metricfile
    
    if np.max(n_range) > nml.n_max:
        raise ValueError('Invalid value : n_range')
    # if (nml.N_t % nml.N_p_n_out) != 0:
    #     print('N_p_n_out of GKNET.data has to be a divisor of N_t')

    # Set 2D grid size for reshaping -------------------------------------------
    nx, ntht = nml.N_x, nml.N_z*nml.z_ratio
    shape2D = (ntht,nx)
    
    grid_file = '../RZ-table.txt'
    R,Z = read_grid(grid_file, shape2D)

    plot_margin = 5
    xlim =[np.min(R)-plot_margin, np.max(R)+plot_margin]
    ylim =[np.min(Z)-plot_margin, np.max(Z)+plot_margin]

    if pmode == 0: title_p =r'$\phi_{1}$'
    if pmode == 1: title_p =r'$A_{\parallel}$'
    xlabel=r'$R-R_0 ~ [\rho_{\rm{i}}]$'
    ylabel=r"$Z ~ [\rho_{\rm{i}}]$"

    # Make magnetic contour ----------------------------------------------------
    R_met, Z_met = [],[]
    if plot_metric:
        if nml.num_eq_flag != 1:
            # analytical equilibria
            a_0 = nml.R_0 * nml.ratio_a
            R_met = np.fromfunction(lambda i, j: a_0*(i/nx)*np.cos(2*np.pi*j/ntht), (nx, ntht+1), dtype=float)
            Z_met = np.fromfunction(lambda i, j: a_0*(i/nx)*np.sin(2*np.pi*j/ntht), (nx, ntht+1), dtype=float)
        elif not os.path.exists(metric_path):
            print('Assigned metric path does not exist')
            plot_metric = False
        else:
            # numerical equilibria
            nss = nml.N_x + nml.N_x_left + nml.N_x_right
            R_met, Z_met = read_metric(nss, ntht, metric_path, nml.R_0)
            xlim =[np.min(R_met)-plot_margin, np.max(R_met)+plot_margin]
            ylim =[np.min(Z_met)-plot_margin, np.max(Z_met)+plot_margin]

    # Plot contour from zarr ---------------------------------------------------
    if is_zarr:
        if pmode == 0: z = zarr.open('phi.zarr', mode='r')
        if pmode == 1: z = zarr.open('apl.zarr', mode='r')

        nxt = len(t_range)*len(n_range)
        for nxt_index in tqdm(range(nxt)):
            t_index = nxt_index // len(n_range)
            n_index = nxt_index % len(n_range)
            t = t_range[t_index]
            n = n_range[n_index]

            title_t = ',  $t ~ v_{t0i}/R_0=$'+f'{(nml.L_t/nml.N_p_n_out)*t:.3} '
            title_n = f',  $n=${n*nml.N_wedge}'
            title = title_p + title_n + title_t
            phi = np.real(z[t,n,:,:])
            if pmode == 0: fname = f'./phi/{n:02d}P{t:04d}.png'
            if pmode == 1: fname = f'./apl/{n:02d}A{t:04d}.png'
            vmax = np.max(np.abs(phi))
            vlim =[-vmax, vmax]
        
            plotPolSlice(phi,R,Z,vlim,xlim,ylim,xticks,yticks,fname,
                            title=title,xlabel=xlabel,ylabel=ylabel,
                            plot_bndry=plot_bndry,color_bg=color_bg,
                            plot_metric=plot_metric,
                            plot_metric_theta = plot_metric_theta,
                            R_met=R_met, Z_met=Z_met,
                            cmap=cmap, figsize=figsize)
    # Plot contour from text ---------------------------------------------------
    else:
        nxt = len(t_range)*len(n_range)
        for nxt_index in tqdm(range(nxt)):
            t_index = nxt_index // len(n_range)
            n_index = nxt_index % len(n_range)
            t = t_range[t_index]
            n = n_range[n_index]

            title_t = ',  $t ~ v_{t0i}/R_0=$'+f'{(nml.L_t/nml.N_p_n_out)*t:.3} '
            title_n = f',  $n=${n*nml.N_wedge}'
            title = title_p + title_n + title_t
            if pmode == 0: phi,fname = read_phi('../phi/'+format(n,'02')+'P'+format(t,'04')+'.txt', shape2D)
            if pmode == 1: phi,fname = read_phi('../apl/'+format(n,'02')+'A'+format(t,'04')+'.txt', shape2D)
            vmax = np.max(np.abs(phi))
            vlim =[-vmax, vmax]
        
            plotPolSlice(phi,R,Z,vlim,xlim,ylim,xticks,yticks,fname,
                            title=title,xlabel=xlabel,ylabel=ylabel,
                            plot_bndry=plot_bndry,color_bg=color_bg,
                            plot_metric=plot_metric,
                            plot_metric_theta = plot_metric_theta,
                            R_met=R_met, Z_met=Z_met,
                            cmap=cmap, figsize=figsize)

def main():
    #***************************************************************************
    #**    USER SETTINGS                                                      **
    #***************************************************************************
    config = Config(
        # General settings ---------------------------------
        version   = 'AB',
        path_root = '..',
        pmode = 0,
        t_range = [10],
        n_range = [7],

        # Plot settings ------------------------------------
        xticks = [100,10],
        yticks = [100,10],
        cmap = 'jet',
        figsize = (12,8),
        color_bg = 'w',
        plot_bndry = False,
        plot_metric = False,

        # Advanced settings --------------------------------
        plot_metric_theta = False,
        is_zarr = False,
    )
    # ###### SETTING #######################################
    # General settings -------------------------------------
    # version        : GKNET version
    # pmode          : 0: phi, 1: A_parallel
    # t_range        : Sequence type, ref **P[****].txt
    # n_range        : Sequence type, ref [**]P****.txt
    #
    # Plot settings ----------------------------------------
    # xticks         : [major locator, minor locator]
    # yticks         : [major locator, minor locator]
    # cmap           : colormap
    # figsize        : figure size
    # color_bg       : background color
    # plot_bndry     : plot boundary with bold line
    # plot_metric    : plot iso-psi (& iso-theta) contour
    #
    # Advanced settings ------------------------------------
    # plot_metric_theta : plot iso-theta contour
    # is_zarr           : zarr input
    # ######################################################
    mode_contour(config)

    
if __name__ == "__main__":
    main()