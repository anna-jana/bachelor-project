"""
This module implements the derivatives of the time with respect to time temperature
dt/dT and d^2t/dT^2
Everything in eV powers
"""

import numpy as np

from config import model
import g_star


def dtdT(T, g_model=g_star.borsamyi_paper_table):
    """
    Computes the derivative dt/dT at T using a model the effective relativistic degrees of
    freedom a GStarModel
    """
    return (
        - model.M_pl * np.sqrt(45 / (64 * np.pi**3)) *
        1 / (T**3 * g_model.g_s(T) * np.sqrt(g_model.g_rho(T))) *
        (T*g_model.g_rho_diff(T) + 4*g_model.g_rho(T))
    )

def d2tdT2(T, g_model=g_star.borsamyi_paper_table):
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
        - model.M_pl * np.sqrt(45 / (64 * np.pi**3)) * (
            - (3 * T**2 * g_s * g_rho**0.5 + T**3 * g_s_diff * g_s**0.5 + g_rho * T**3 * g_s / (2 * g_s**0.5)) /
              (T**3 * g_s * g_s**0.5)**2 *
              (T * g_s_diff + 4 * g_s)
            + (g_rho_diff + T * g_rho_diff2 + 4 * g_rho_diff) /
              (T**3 * g_s * g_rho**0.5)
        )
    )
