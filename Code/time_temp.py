"""
This module implements the derivatives of the time with respect to time temperature
dt/dT and d^2t/dT^2
Everything in eV powers
"""

import numpy as np
import scipy.optimize as opt

from config import model
import g_star

def dtdT(T, g_model, parameter=model):
    """
    Computes the derivative dt/dT at T using a model the effective relativistic degrees of
    freedom a GStarModel
    """
    return (
        - np.sqrt(8*np.pi) * parameter.M_pl * np.sqrt(45 / (64 * np.pi**3)) *
        1 / (T**3 * g_model.g_s(T) * np.sqrt(g_model.g_rho(T))) *
        (T*g_model.g_rho_diff(T) + 4*g_model.g_rho(T))
    )

def d2tdT2(T, g_model, parameter=model):
    """
    Computes the derivative d^2t/dT^2 at T using a model the effective relativistic degrees of
    freedom a GStarModel
    """
    g_s = g_model.g_s(T)
    g_s_diff = g_model.g_s_diff(T)
    g_rho = g_model.g_rho(T)
    g_rho_diff = g_model.g_rho_diff(T)
    g_rho_diff2 = g_model.g_rho_diff2(T)
    return (
        - np.sqrt(8*np.pi) * parameter.M_pl * np.sqrt(45 / (64 * np.pi**3)) * (
            - (3 * T**2 * g_s * g_rho**0.5 + T**3 * g_s_diff * g_rho**0.5 + T**3 * g_s * g_rho_diff / (2 * g_rho**0.5)) /
              (T**3 * g_s * g_rho**0.5)**2 *
              (T * g_rho_diff + 4 * g_rho)
            + (g_rho_diff + T * g_rho_diff2 + 4 * g_rho_diff) /
              (T**3 * g_s * g_rho**0.5)
        )
    )


def hubble_parameter_in_rad_epoch(T, g_model, parameter=model):
    """
    Compute Hubble parameter in the radiation dominated epoch using the
    Friedmann equation 3 H^2 M_pl^2 = rho = pi^2 / 30 * g_* T^4
    from the temperature T in eV
    """
    rho = np.pi**2 / 30 * g_model.g_rho(T) * T**4
    H = np.sqrt(rho / (3 * parameter.M_pl**2))
    return H

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
    sol = opt.root(lambda T_guess: N * hubble_parameter_in_rad_epoch(T_guess, g_model) - m_a_fn(T_guess, f_a), T_init_guess)
    # assert sol.success, "%f MeV" % (sol.x / 1e6)
    T_osc = sol.x
    return T_osc[0]
