from dataclasses import dataclass
import f90nml
import numpy as np

"""##################################################################################
    read_gknet_nml.py

    ***EXPLANATION***
    read namelist GKNET.data

    ***VERSION***
    v1.1 (2026/07)
    - add version 'AB' (default value)
    - add new arguments: version, N_spec 
    v2.0 (2026/11)
    - new function 'read_metric' to input metric file
    - add new arguments: energy_out_time, contour_out_time

##################################################################################"""

@dataclass
class SimpleNameList:
    version: str

    N_spec: int

    nonlinear: bool
    R_0: float
    beta_i: float
    ratio_a: float

    N_mode_z_out: int
    N_p_n_out: int
    time_start_omega: float
    energy_out_time: float
    contour_out_time: float

    L_t: float
    N_t: int

    N_wedge: int
    N_x_left: int
    N_x_right: int
    N_x: int
    N_y: int
    N_z: int
    N_v: int
    N_u: int

    z_ratio: int
    n_max: int

    num_eq_flag: int
    param_flag: int
    metricfile: str
    paramfile: str


@dataclass
class MetricData:
    rho  : np.ndarray
    theta: np.ndarray

    psi_sq: np.ndarray
    phi_sq: np.ndarray
    vol_sq: np.ndarray

    R: np.ndarray
    Z: np.ndarray

    g_rr_dn: np.ndarray
    g_rp_dn: np.ndarray
    g_tr_dn: np.ndarray
    g_pp_dn: np.ndarray
    g_pt_dn: np.ndarray
    g_tt_dn: np.ndarray

    g_rr_up: np.ndarray
    g_rp_up: np.ndarray
    g_tr_up: np.ndarray
    g_pp_up: np.ndarray
    g_pt_up: np.ndarray
    g_tt_up: np.ndarray

    B_p_up: np.ndarray
    B_t_up: np.ndarray



def simple_gknet_nml(path='../GKNET.data', version='AB'):
    if version == 'AB':
        gd = f90nml.read(path)
        L_t              = gd['PARAM_TIME']['L_t']
        energy_out_time  = gd['PARAM_TIME']['energy_out_time']
        contour_out_time = gd['PARAM_TIME']['contour_out_time']
        N_mode_z_out     = int(L_t//energy_out_time)
        N_p_n_out        = int(L_t//contour_out_time)

        nml = SimpleNameList(
            version = version,

            N_spec    = gd['Species']['N_spec'],

            nonlinear = gd['PARAM_CONSTANT']['nonlinear'],
            R_0       = gd['PARAM_CONSTANT']['R_0'],
            beta_i    = gd['PARAM_CONSTANT']['beta'],
            ratio_a   = gd['PARAM_CONSTANT']['ratio_a'],

            N_mode_z_out     = N_mode_z_out,
            N_p_n_out        = N_p_n_out,
            time_start_omega = gd['PARAM_TIME']['time_start_omega'],
            L_t              = gd['PARAM_TIME']['L_t'],
            N_t              = gd['PARAM_TIME']['N_t'],

            energy_out_time = energy_out_time,
            contour_out_time = contour_out_time,

            N_wedge   = gd['PARAM_GRID']['N_w'],
            N_x_left  = gd['PARAM_GRID']['N_x_left'],
            N_x_right = gd['PARAM_GRID']['N_x_right'],
            N_x       = gd['PARAM_GRID']['N_x'],
            N_y       = gd['PARAM_GRID']['N_y'],
            N_z       = gd['PARAM_GRID']['N_z'],
            N_v       = gd['PARAM_GRID']['N_v'],
            N_u       = gd['PARAM_GRID']['N_u'],
            z_ratio   = gd['PARAM_GRID']['z_ratio'],
            n_max     = gd['PARAM_GRID']['n_max'],

            num_eq_flag = gd['PARAM_EQ']['num_eq_flag'],
            metricfile  = gd['PARAM_EQ']['metricfile'],

            param_flag  = gd['PARAM_PF']['param_flag'],
            paramfile   = gd['PARAM_PF']['paramfile'],
        )

    if version == 'PS':
        gd = f90nml.read(path)
        nml = SimpleNameList(
            version = version,

            N_spec    = gd['Species']['N_spec'],

            nonlinear = gd['PARAM_CONSTANT']['nonlinear'],
            R_0       = gd['PARAM_CONSTANT']['R_0'],
            beta_i    = gd['PARAM_CONSTANT']['beta'],
            ratio_a   = gd['PARAM_CONSTANT']['ratio_a'],

            N_mode_z_out     = gd['PARAM_OUTPUT']['N_mode_z_out'],
            N_p_n_out        = gd['PARAM_OUTPUT']['N_p_n_out'],
            time_start_omega = gd['PARAM_OUTPUT']['time_start_omega'],

            L_t = gd['PARAM_TIME']['L_t'],
            N_t = gd['PARAM_TIME']['N_t'],

            N_wedge   = gd['PARAM_GRID']['N_w'],
            N_x_left  = gd['PARAM_GRID']['N_x_left'],
            N_x_right = gd['PARAM_GRID']['N_x_right'],
            N_x       = gd['PARAM_GRID']['N_x'],
            N_y       = gd['PARAM_GRID']['N_y'],
            N_z       = gd['PARAM_GRID']['N_z'],
            N_v       = gd['PARAM_GRID']['N_v'],
            N_u       = gd['PARAM_GRID']['N_u'],
            z_ratio   = gd['PARAM_GRID']['z_ratio'],
            n_max     = gd['PARAM_GRID']['n_max'],

            num_eq_flag = gd['PARAM_EQ']['num_eq_flag'],
            param_flag  = gd['PARAM_EQ']['param_flag'],
            metricfile  = gd['PARAM_EQ']['metricfile'],
            paramfile   = gd['PARAM_EQ']['paramfile'],
        )
    
    return nml


def read_metric(root='..', version='AB'):
    #=============================================
    # Note that length and magnetic field are normalized by R_0 and B_0, respectively
    # g_up has L^2, g_down has L^-2 dimension
    #=============================================

    nml_tmp = simple_gknet_nml(f'{root}/GKNET.data', version)
    if nml_tmp.num_eq_flag != 1:
        print('This function is only for numerical equilibrium!')
        return None

    nss  = nml_tmp.N_x_left + nml_tmp.N_x + nml_tmp.N_x_right
    ntht = nml_tmp.N_z * nml_tmp.z_ratio

    data = np.loadtxt(f'{root}/{nml_tmp.metricfile}', dtype=float)
    data = np.reshape(data, (nss, ntht, -1))

    mtr = MetricData(
        rho   = data[:,0,0],
        theta = data[0,:,1],

        psi_sq = data[:,0,19],
        phi_sq = data[:,0,20],
        vol_sq = data[:,0,21],

        R = data[:,:,17],
        Z = data[:,:,18],

        g_rr_dn = data[:,:,3],
        g_rp_dn = data[:,:,4],
        g_tr_dn = data[:,:,5],
        g_pp_dn = data[:,:,6],
        g_pt_dn = data[:,:,7],
        g_tt_dn = data[:,:,8],

        g_rr_up = data[:,:,9],
        g_rp_up = data[:,:,10],
        g_tr_up = data[:,:,11],
        g_pp_up = data[:,:,12],
        g_pt_up = data[:,:,13],
        g_tt_up = data[:,:,14],

        B_p_up = data[:,:,15],
        B_t_up = data[:,:,16],
    )

    return mtr


if __name__ == '__main__':
    simple_gknet_nml()