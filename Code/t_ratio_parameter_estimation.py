import sys
import copy
import math

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import emcee
import corner

import axion_mass
import potential
import g_star
import eom
import config
import time_temp

def log_gaussian(x, x_mean, x_stdev):
    return - (x - x_mean)**2 / (2 * x_stdev**2) - 0.5 * np.log(2 * np.pi * x_stdev**2)

def compute_Omega(theta_i, f_a, T_ratio, mu, zeta, debug_print=True):
    if debug_print:
        print("*", end="")
        sys.stdout.flush()
    g = g_star.make_micro_from_T_ratio(T_ratio, mu)
    T_i = time_temp.find_T_osc(f_a, lambda T, f: axion_mass.micro_m_a(T, f, mu, zeta), g)
    if T_i <= config.parameter.T_eq:
        print("WARNING: oscillation starts below T_eq")
        return np.nan
    T_f = config.parameter.T0
    m_i = axion_mass.micro_m_a(T_ratio * T_i, f_a, mu, zeta)
    m_f = axion_mass.micro_m_a(T_ratio * T_f, f_a, mu, zeta)
    g_s_f = g.g_s(T_f)
    g_s_i = g.g_s(T_i)
    A_i = theta_i * f_a
    rho = 0.5 * m_i * m_f * A_i**2 * g_s_f / g_s_i * (T_f / T_i)**3
    Omega = rho / config.parameter.rho_c * config.parameter.h**2
    return Omega

Delta_N_eff_BBM_mean = 3.28 - 3
Delta_N_eff_BBM_err = 0.28
Delta_N_eff_CMB_mean = 3 - 2.99
Delta_N_eff_CMB_err = 0.3
T_BBM = 1e6
T_CMB = config.parameter.T0

def ln_likelihood(THETA):
    theta_i, log_f_a, T_ratio, mu, zeta = THETA
    f_a = 10**log_f_a
    density_parameter_computed = compute_Omega(theta_i, f_a, T_ratio, mu, zeta)
    Delta_N_eff_at_BBM = g_star.compute_Delta_N_eff(T_ratio, mu, T_BBM)
    Delta_N_eff_at_CMB = g_star.compute_Delta_N_eff(T_ratio, mu, T_CMB)
    return (
        log_gaussian(density_parameter_computed, config.parameter.Omega_DM_h_sq, config.parameter.Omega_DM_h_sq_err) +
        log_gaussian(Delta_N_eff_at_BBM, Delta_N_eff_BBM_mean, Delta_N_eff_BBM_err) +
        log_gaussian(Delta_N_eff_at_CMB, Delta_N_eff_CMB_mean, Delta_N_eff_CMB_err)
    )

parameter_names = ["theta_i", "log f_a", "T_ratio", "mu", "zeta"]

zeta_mean = 1.34
zeta_err = 0.01

def make_initial_guess():
    theta_i = np.random.uniform(0, np.pi)
    log_f_a = np.random.uniform(15, 18) + 9
    mu = 10**np.random.uniform(2, 3)
    T_ratio = 1 / np.random.uniform(3, 5)
    zeta = np.random.uniform(zeta_mean - zeta_err, zeta_mean + zeta_err)
    return (theta_i, log_f_a, T_ratio, mu, zeta)

def ln_prior(THETA):
    theta_i, log_f_a, T_ratio, mu, zeta = THETA
    if 0 < theta_i <= np.pi and 15 <= log_f_a - 9 <= 18 and \
       2 <= np.log10(mu) <= 3 and abs(zeta - zeta_mean) <= zeta_err and 1 / 5 <= T_ratio <= 1 / 3:
        return 0
    else:
        return - np.inf # = log 0

def ln_prob(THETA):
    lp = ln_prior(THETA)
    like = ln_likelihood(THETA)
    if not np.isfinite(lp) or not np.isfinite(like):
        return -np.inf
    return lp + like

ndim = 2 + 3
num_walkers = ndim * 2 * 5 * 2 * 2
steps = 5000 * 4 * 4
num_threads = 4

if __name__ == "__main__":
    pos = [make_initial_guess() for i in range(num_walkers)]
    sampler = emcee.EnsembleSampler(num_walkers, ndim, ln_prob, threads=num_threads)
    sampler.run_mcmc(pos, steps)
    np.savez(config.data_path + "/T_ratio_micro_qcd_parameter.npz", samples=sampler.chain)
