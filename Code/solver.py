import ctypes

import numpy as np
import solver
import time_temp
import axion_mass
import g_star
import potential
import eom
import config

class Parameter(ctypes.Structure):
    _fields_ = [
        ("M_pl", ctypes.c_double),
        ("Lambda_QCD", ctypes.c_double),
        ("T0", ctypes.c_double),
        ("rho_c", ctypes.c_double),
        ("m_u", ctypes.c_double),
        ("m_d", ctypes.c_double),
        ("m_pi0", ctypes.c_double),
        ("f_pi0", ctypes.c_double),
    ]

libsolver = ctypes.CDLL("./libsolver.so")
libsolver.solver.argtypes = (Parameter, ctypes.c_double, ctypes.c_double, ctypes.c_double)
libsolver.solver.restype = ctypes.c_double

def compute_relic_density(parameter, theta_i, f_a):
    T_osc = time_temp.find_T_osc(f_a, axion_mass.m_a_from_chi_general, g_star.matched)
    ps = Parameter(parameter.M_pl, parameter.Lambda_QCD, parameter.T0, parameter.rho_c, parameter.m_u, parameter.m_d, parameter.m_pi0, parameter.f_pi0)
    global libsolver
    ans = libsolver.solver(ps, ctypes.c_double(T_osc), ctypes.c_double(theta_i), ctypes.c_double(f_a))
    return float(ans)

