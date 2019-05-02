"""
This module provides the g_star quantity as a numerical function.
It is given as data points as well as a fitting interpolation (Pchip)
The temperature is given in eV.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

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

# g_s_data = np.loadtxt("g_s_data.dat")
# g_s_over_g_rho_data = np.loadtxt("g_rho_over_g_s.dat")
#
# T_g_s = g_s_data[:, 0] * 1e6 # [eV]
# g_s = g_s_data[:, 1]
#
# T_g_rho_over_g_s = g_s_over_g_rho_data[:, 0] * 1e6
# T_g_rho = T_g_rho_over_g_s
# g_rho_over_g_s = g_s_over_g_rho_data[:, 1]
# g_rho = g_rho_over_g_s * g_s_interp(T_g_rho_over_g_s)
# log10_T_per_MeV = data[:, 0]
#
# g_rho_from_plot = g_rho
# g_s_from_plot = g_s

g_rho = data[:, 1]
g_rho_per_s = data[:, 2]

MeV = 1e6 # [eV]
T = MeV * 10 ** log10_T_per_MeV # [eV]
g_s = g_rho / g_rho_per_s # [1]

# T = np.concatenate([T, T_g_rho])
# sort_perm = np.argsort(T)
# T =

g_rho_interp = PchipInterpolator(T, g_rho)
g_s_interp = PchipInterpolator(T, g_s)

g_rho_interp_diff = g_rho_interp.derivative()
g_s_interp_diff = g_s_interp.derivative()
