"""
This module implements the derivatives of the time with respect to time temperature
dt/dT and d^2t/dT^2
Everything in eV powers
"""

import numpy as np

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


