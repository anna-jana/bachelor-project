import ctypes

import numpy as np
import solver
import time_temp
import axion_mass
import g_star
import potential
import eom
import config
from numpy import sqrt


libsolver = ctypes.CDLL("./libsolver2.so")
libsolver.solver.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double)
libsolver.solver.restype = ctypes.c_double



def compute_relic_density(Delta_N_eff, mu, micro_quark_mass, zeta, N_f, N_mu, kappa, c_N, eta, m_psi, theta_i, f_a):
    T_osc = time_temp.find_T_osc(f_a, lambda T, f_a: axion_mass.micro_m_a(T, f_a, mu, zeta), g_star.matched) # hopefully that works
    global libsolver
    ans = libsolver.solver(
             ctypes.c_double(Delta_N_eff),
             ctypes.c_double(mu),
             ctypes.c_double(micro_quark_mass),
             ctypes.c_double(zeta),
             ctypes.c_double(N_f),
             ctypes.c_double(N_mu),
             ctypes.c_double(kappa),
             ctypes.c_double(c_N),
             ctypes.c_double(eta),
             ctypes.c_double(m_psi),
            ctypes.c_double(T_osc), ctypes.c_double(theta_i), ctypes.c_double(f_a))
    return float(ans)

