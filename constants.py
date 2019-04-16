"""
This file contains parameters and constants that are used
in a lot of different places.
All of them are in natural units with
epsilon_0 = k_B = c = hbar = 1
"""

import numpy as np
from astropy import units as u, constants as ac
from scipy import constants as sc

Lambda_QCD = 200 * 1e6
H0 = ((67.74 * u.km / u.second / u.Mpc * ac.hbar).to("eV") / u.eV).to_value()
# TODO: those two planck masses differ
M_pl = 2.435e18 * 1e9 # [eV]
# M_p = sc.physical_constants["Planck mass"][0] * sc.c**2 / sc.elementary_charge
h = 0.67
rho_c = 1 / (3 * H0**2 * M_pl**2)
z_eq = 3400
G = 6.70861e-39
Omega_DM_h_sq = 0.12
Omega_c = Omega_DM_h_sq / h**2
Omega_b = 0.022 / h**2
Omega_m = Omega_c + Omega_b
Omega_rad = Omega_m / (z_eq + 1)
T0 = 2.73 * sc.Boltzmann / sc.elementary_charge
