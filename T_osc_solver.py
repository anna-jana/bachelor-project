"""
This model implements a numerical solution to the temperture T_osc at which the axion field
starts to oscillate.
"""

import numpy as np
import scipy.optimize as opt

import time_temp


# N H(T) = m_a(T) ===> root of N H(T) - m_a(T)
def find_T_osc(f_a, m_a_fn, g_model, N=3):
    """
    Computes the temperature at which the axion field starts to oscillate.
    Take the axion decay constant f_a in eV, the mass in eV as a python function m_a_fn : T x f_a -> m_a, and the
    factor N=3 for the condition N H(T_osc) = m_a(T_osc)
    Returns the temperature T_osc in eV
    """
    # T_init_guess = (30 / np.pi**2 * axion_mass.m_a_at_abs_zero_from_shellard(f_a) / N) ** (1 / 4) # g = 1
    T_init_guess = 1e9
    # print(T_init_guess)
    sol = opt.root(lambda T_guess: N * time_temp.hubble_parameter_in_rad_epoch(T_guess, g_model) - m_a_fn(T_guess, f_a), T_init_guess)
    # assert sol.success, "%f MeV" % (sol.x / 1e6)
    T_osc = sol.x
    return T_osc
