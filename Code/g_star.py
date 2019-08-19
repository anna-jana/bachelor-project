"""
This module provides the g_star quantity as a numerical function.
It is given as data points as well as a fitting interpolation (Pchip)
The temperature is given in eV.
"""

from collections import namedtuple
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

import config

MeV = 1e6 # [eV]

GStarModel = namedtuple("GStarModel", ["g_rho", "g_s", "g_rho_diff", "g_s_diff", "g_rho_diff2"])

############################# g data from the table in the paper Borsamyi #######################
# data from lattice paper (S4.3) from Borsamyi et al.
# columns: [log10(T/MeV), g_rho, g_rho / g_s]
data = np.array([
    [0, 10.71, 1.00228],
    [0.5, 10.74, 1.00029],
    [1.0, 10.76, 1.00048],
    [1.25, 11.09, 1.00505],
    [1.6, 13.68, 1.02159],
    [2, 17.61, 1.02324],
    [2.15, 24.07, 1.05423],
    [2.2, 29.84, 1.07578],
    [2.4, 47.83, 1.06118],
    [2.5, 53.04, 1.0469],
    [3, 73.48, 1.01778],
    [4, 83.1, 1.00123],
    [4.3, 85.56, 1.00389],
    [4.6, 91.97, 1.00887],
    [5, 102.17, 1.0075],
    [5.45, 104.98, 1.00023],
])

# convert data from table in paper
log10_T_per_MeV = data[:, 0]
g_rho = data[:, 1]
g_rho_per_s = data[:, 2]
T = MeV * 10 ** log10_T_per_MeV # [eV]
g_s = g_rho / g_rho_per_s # [1]
# interpolate
g_rho_interp = PchipInterpolator(T, g_rho)
g_s_interp = PchipInterpolator(T, g_s)

borsamyi_paper_table = GStarModel(g_rho=g_rho_interp, g_s=g_s_interp,
        g_rho_diff=g_rho_interp.derivative(), g_s_diff=g_s_interp.derivative(), g_rho_diff2=g_rho_interp.derivative(2))

g_s_paper = g_s
g_rho_paper = g_rho

## g data extracted from plots + table from paper
# load extracted data from files
g_s_data = np.loadtxt(config.data_path + "/g_s_data.dat")
g_s_over_g_rho_data = np.loadtxt(config.data_path + "/g_rho_over_g_s.dat")

# convert g_s data and combine with table
T_g_s = np.concatenate([g_s_data[:, 0] * MeV, T])
g_s = np.concatenate([g_s_data[:, 1], g_s])
sort_perm = np.argsort(T_g_s)
T_g_s = T_g_s[sort_perm]
g_s = g_s[sort_perm]

# remove bad indicies (found by hand)
g_s_bad_indicies = [1,2,3,4,6,7,8,11,13,15,20]
g_s = np.delete(g_s, g_s_bad_indicies)
T_g_s = np.delete(T_g_s, g_s_bad_indicies)

# interpolate g_s
g_s_interp = PchipInterpolator(T_g_s, g_s)

# convert ratio data
T_g_rho_over_g_s = g_s_over_g_rho_data[:, 0] * MeV
g_rho_over_g_s = g_s_over_g_rho_data[:, 1]

# comptute g_rho
T_g_rho = np.concatenate([T_g_rho_over_g_s, T])
g_rho = np.concatenate([g_rho_over_g_s * g_s_interp(T_g_rho_over_g_s), g_rho])
sort_perm = np.argsort(T_g_rho)
T_g_rho = T_g_rho[sort_perm]
g_rho = g_rho[sort_perm]

# remove bad indicies (found by hand)
g_rho_bad_indicies = [1,2,3,4,5,7,8,9,10]
g_rho = np.delete(g_rho, g_rho_bad_indicies)
T_g_rho = np.delete(T_g_rho, g_rho_bad_indicies)

# interpolate g_rho
g_rho_interp = PchipInterpolator(T_g_rho, g_rho)

# make model
borsamyi_table = GStarModel(g_rho=g_rho_interp, g_s=g_s_interp,
        g_rho_diff=g_rho_interp.derivative(), g_s_diff=g_s_interp.derivative(), g_rho_diff2=g_rho_interp.derivative(2))



############################################### Shellard et al. fit #############################################
a0 = np.array([1.21, 1.36])
a = np.array([
    [[0.572, 0.33, 0.579, 0.138, 0.108],
     [-8.77, -2.95, -1.8, -0.162, 3.76],
     [0.682, 1.01, 0.165, 0.934, 0.869]],
    [[0.498, 0.327, 0.579, 0.14, 0.109],
     [-8.74, -2.89, -1.79, -0.102, 3.82],
     [0.693, 1.01, 0.155, 0.963, 0.907]],
])

def g(T, i):
    t = np.log(T / 1e9)
    return np.exp(a0[i] + np.sum(a[i, 0, :] * (1.0 + np.tanh((t - a[i, 1, :]) / a[i, 2, :]))))

def sech(x): return 1 / np.cosh(x)

def dgdT(T, i):
    t = np.log(T / 1e9)
    x = (t - a[i, 1, :]) / a[i, 2, :]
    return np.sum(a[i, 0, :] / a[i, 2, :] * sech(x)**2 / T) * g(T, i)

def d2gdT2(T, i):
    t = np.log(T / 1e9)
    x = (t - a[i, 1, :]) / a[i, 2, :]
    return np.sum(a[i, 0, :] / a[i, 2, :] * sech(x)**2 / T**2 * g(T, i) * (
        -2 * np.tanh(x) / a[i, 2, :] + a[i, 0, :] / a[i, 2, :] * sech(x)**2 - 1
    ))

def vec(f):
    ufunc = np.frompyfunc(f, 1, 1)
    return lambda T: ufunc(T).astype("float")

shellard_fit = GStarModel(g_rho=vec(lambda T: g(T, 0)), g_s=vec(lambda T: g(T, 1)), g_rho_diff=vec(lambda T: dgdT(T, 0)),
        g_s_diff=vec(lambda T: dgdT(T, 1)), g_rho_diff2=vec(lambda T: d2gdT2(T, 0)))


##################################################### matched result ##################################################
# match shellard and bosamyi
T_min = 1e6

def match_g_rho(T):
    if T < T_min:
        return borsamyi_table.g_rho(T_min) / shellard_fit.g_rho(T_min) * shellard_fit.g_rho(T)
    else:
        return borsamyi_table.g_rho(T)

def match_g_s(T):
    if T < T_min:
        return borsamyi_table.g_s(T_min) / shellard_fit.g_s(T_min) * shellard_fit.g_s(T)
    else:
        return borsamyi_table.g_s(T)

def match_dg_rhodT(T):
    if T < T_min:
        return borsamyi_table.g_rho_diff(T_min) / shellard_fit.g_rho_diff(T_min) * shellard_fit.g_rho_diff(T)
    else:
        return borsamyi_table.g_rho_diff(T)

def match_dg_sdT(T):
    if T < T_min:
        return borsamyi_table.g_s_diff(T_min) / shellard_fit.g_s_diff(T_min) * shellard_fit.g_s_diff(T)
    else:
        return borsamyi_table.g_s_diff(T)

def match_d2g_rhoT2(T):
    if T < T_min:
        return borsamyi_table.g_rho_diff2(T_min) / shellard_fit.g_rho_diff2(T_min) * shellard_fit.g_rho_diff2(T)
    else:
        return borsamyi_table.g_rho_diff2(T)

matched = GStarModel(g_rho=vec(match_g_rho), g_s=vec(match_g_s),
        g_rho_diff=vec(match_dg_rhodT), g_s_diff=vec(match_dg_sdT), g_rho_diff2=vec(match_d2g_rhoT2))

def T_neutrino(T):
    if(T < 1e-2 * 1e6):
        return pow(4.0 / 11, 1/3.) * T;
    else:
        f = pow(8*(matched.g_s(T) - 2) / (7.*6), 1./3);
        if(f < 1.0):
            return f * T;
        else:
            return T;

def make_micro(Delta_N_eff):
    def d_T_nu_dT(T):
        if(T < 1e-2 * 1e6):
            return pow(4.0 / 11, 1/3.);
        else:
            f = pow(8*(matched.g_s(T) - 2) / (7.*6), 1./3);
            if(f < 1.0):
                return 1 / 3.0 * pow(8 / 7. / 6. * matched.g_s_diff(T), 1. / 3 - 1) * T + f;
            else:
                return 1.0;
    micro = GStarModel(
            g_rho = lambda T: matched.g_rho(T) + Delta_N_eff * 7 / 8. * pow(T_neutrino(T) / T, 4),
            g_s = lambda T: matched.g_s(T) + Delta_N_eff * 7 / 8. * pow(T_neutrino(T) / T, 3),
            g_rho_diff = lambda T: matched.g_rho_diff(T) + Delta_N_eff * 7 / 8. * 4 * pow(T_neutrino(T) / T, 4 - 1) * (d_T_nu_dT(T) / T - T_neutrino(T) / (T*T)),
            g_s_diff = lambda T: matched.g_s_diff(T) + Delta_N_eff * 7 / 8. * 3 * pow(T_neutrino(T) / T, 3 - 1) * (d_T_nu_dT(T) / T - T_neutrino(T) / (T*T)),
            g_rho_diff2 = matched.g_rho_diff2,
    )
    return micro

N_prime = 31

def compute_Delta_N_eff(T_ratio, mu, T):
    a = 5e-2
    N_prime = (31 - 3.5) / 2 * np.tanh(a * T - a * mu) + (31 + 3.5) / 2
    return 8 / 7 * (T_ratio * T / T_neutrino(T))**4 * N_prime

def make_micro_from_T_ratio(T_ratio, mu):
    def d_T_nu_dT(T):
        if(T < 1e-2 * 1e6):
            return pow(4.0 / 11, 1/3.);
        else:
            f = pow(8*(matched.g_s(T) - 2) / (7.*6), 1./3);
            if(f < 1.0):
                return 1 / 3.0 * pow(8 / 7. / 6. * matched.g_s_diff(T), 1. / 3 - 1) * T + f;
            else:
                return 1.0;
    micro = GStarModel(
            g_rho = lambda T: matched.g_rho(T) + compute_Delta_N_eff(T_ratio, mu, T) * 7 / 8. * pow(T_neutrino(T) / T, 4),
            g_s = lambda T: matched.g_s(T) + compute_Delta_N_eff(T_ratio, mu, T) * 7 / 8. * pow(T_neutrino(T) / T, 3),
            g_rho_diff = lambda T: matched.g_rho_diff(T) + compute_Delta_N_eff(T_ratio, mu, T) * 7 / 8. * 4 * pow(T_neutrino(T) / T, 4 - 1) * (d_T_nu_dT(T) / T - T_neutrino(T) / (T*T)),
            g_s_diff = lambda T: matched.g_s_diff(T) + compute_Delta_N_eff(T_ratio, mu, T) * 7 / 8. * 3 * pow(T_neutrino(T) / T, 3 - 1) * (d_T_nu_dT(T) / T - T_neutrino(T) / (T*T)),
            g_rho_diff2 = matched.g_rho_diff2,
    )
    return micro
